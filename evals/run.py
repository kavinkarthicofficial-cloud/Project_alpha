"""Memory eval runner.

Each case tells the assistant facts in one or more earlier sessions, then asks a
question in a brand-new session. Passing requires real cross-session memory.

    python -m evals.run --model llama3
    python -m evals.run --model opus --category update --limit 5
    python -m evals.run --model llama3 --no-memory      # baseline: memory turned off
    python -m evals.run --model llama3 --same-session   # upper bound: facts stay in context

Grading: an LLM judge compares each answer with the case's reference answer
(`--judge <alias>`, default: the `judge` route). `--grader substring` is the old,
free but noisy check against `expect_any`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from core.agent.orchestrator import Orchestrator, TurnError
from core.bootstrap import build_orchestrator
from core.config import get_settings
from core.llm.complete import complete
from core.llm.registry import ModelRouter
from core.llm.types import Message, TextDelta

HERE = Path(__file__).parent

JUDGE_SYSTEM = (
    "You grade answers from a personal assistant that is supposed to remember what the "
    "user told it in earlier conversations. You only judge whether the answer conveys "
    "the reference correctly."
)

JUDGE_PROMPT = """\
Question the user asked: {question}

What a correct answer must convey: {reference}

The assistant's answer:
<answer>
{answer}
</answer>

Does the answer convey the reference correctly? Different wording or extra helpful \
detail is fine. A wrong, missing, hedged-into-uselessness or contradictory answer is \
not. Reply with exactly one word: PASS or FAIL."""


@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    answer: str
    error: str | None
    seconds: float
    grader: str
    substring_passed: bool


async def _say(orch: Orchestrator, session_id, text: str, model: str | None) -> tuple[str, str | None]:
    parts: list[str] = []
    error = None
    async for ev in orch.run_turn(session_id, text, model=model):
        if isinstance(ev, TextDelta):
            parts.append(ev.text)
        elif isinstance(ev, TurnError):
            error = ev.message
    return "".join(parts), error


async def judge(router: ModelRouter, preferred: str | None, case: dict, answer: str) -> tuple[bool, str]:
    prompt = JUDGE_PROMPT.format(question=case["question"], reference=case["reference"], answer=answer)
    verdict, profile = await complete(
        router,
        "judge",
        [Message.text("system", JUDGE_SYSTEM), Message.text("user", prompt)],
        preferred=preferred,
    )
    match = re.search(r"\b(PASS|FAIL)\b", verdict.upper())
    return bool(match and match.group(1) == "PASS"), profile.alias


async def run_case(case: dict, args: argparse.Namespace) -> CaseResult:
    start = time.monotonic()
    # Fresh stores per case so memories can't leak between cases.
    orch = build_orchestrator(get_settings(), in_memory=True, memory_enabled=not args.no_memory)
    session = await orch.store.create_session(args.model)

    def result(passed: bool, answer: str, error: str | None, grader: str, sub: bool) -> CaseResult:
        return CaseResult(
            case["id"], case["category"], passed, answer, error, time.monotonic() - start, grader, sub
        )

    for i, turns in enumerate(case["sessions"]):
        if i > 0 and not args.same_session:
            session = await orch.store.create_session(args.model)
        for text in turns:
            _, error = await _say(orch, session.id, text, args.model)
            if error:
                return result(False, "", error, "-", False)
    if not args.same_session:
        session = await orch.store.create_session(args.model)
    answer, error = await _say(orch, session.id, case["question"], args.model)
    if error:
        return result(False, answer, error, "-", False)

    substring_passed = any(e in answer.lower() for e in case["expect_any"])
    if args.grader == "substring":
        return result(substring_passed, answer, None, "substring", substring_passed)
    passed, judge_alias = await judge(orch.router, args.judge, case, answer)
    return result(passed, answer, None, judge_alias, substring_passed)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(HERE / "datasets" / "memory_v0.jsonl"))
    parser.add_argument("--model", help="model alias answering the questions; default: chat route")
    parser.add_argument("--judge", help="model alias grading the answers; default: judge route")
    parser.add_argument("--grader", choices=["judge", "substring"], default="judge")
    parser.add_argument("--category")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--same-session", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    args = parser.parse_args()

    cases = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
    cases = cases[: args.limit] if args.limit else cases

    results: list[CaseResult] = []
    for case in cases:
        r = await run_case(case, args)
        results.append(r)
        mark = "PASS" if r.passed else ("ERR " if r.error else "FAIL")
        print(f"{mark} {r.id:14} {r.seconds:5.1f}s  {(r.error or r.answer).strip()[:90]!r}", flush=True)

    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r.passed)
    print()
    for cat, passes in sorted(by_cat.items()):
        print(f"{cat:12} {sum(passes)}/{len(passes)}")
    total = sum(r.passed for r in results)
    print(f"{'overall':12} {total}/{len(results)}")
    graded = [r for r in results if r.grader not in ("-", "substring")]
    if graded:
        agree = sum(r.passed == r.substring_passed for r in graded)
        print(f"judge ({graded[0].grader}) agreed with substring grading on {agree}/{len(graded)}")
        registry = build_orchestrator(get_settings(), in_memory=True).router.registry
        if any(registry.get(r.grader).local for r in graded):
            print(
                "WARNING: graded by a small local model, which passes wrong answers. "
                "Treat these scores as unreliable; use --judge opus (or another strong model)."
            )

    mode = "same-session" if args.same_session else ("no-memory" if args.no_memory else "memory")
    model_name = args.model or "default"
    out = HERE / "reports" / f"{time.strftime('%Y%m%d-%H%M%S')}-{model_name}-{mode}.json"
    out.write_text(
        json.dumps(
            {
                "model": model_name,
                "mode": mode,
                "grader": args.grader,
                "dataset": Path(args.dataset).name,
                "score": {"passed": total, "total": len(results)},
                "by_category": {c: {"passed": sum(p), "total": len(p)} for c, p in by_cat.items()},
                "results": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    print(f"\nreport: {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    asyncio.run(main())
