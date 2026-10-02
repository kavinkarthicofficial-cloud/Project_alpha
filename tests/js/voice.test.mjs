// Run with: node --test tests/js/*.test.mjs
import assert from "node:assert/strict";
import { test } from "node:test";

import { ClapDetector, matchWakeWord, speechChunks } from "../../apps/api/static/voice.js";

const FRAME_MS = 10;

// Background noise, with sounds placed at given times. Each sound is a list of
// per-frame peak levels.
function signal(totalMs, sounds) {
  const frames = Array.from({ length: totalMs / FRAME_MS }, () => 0.01);
  for (const [atMs, shape] of sounds) {
    shape.forEach((level, i) => {
      frames[atMs / FRAME_MS + i] = level;
    });
  }
  return frames;
}

const CLAP = [0.85, 0.35, 0.08, 0.02];
const SPEECH = Array.from({ length: 30 }, (_, i) => 0.45 + 0.1 * Math.sin(i)); // 300 ms of voice

function events(frames, detector = new ClapDetector()) {
  return frames
    .map((level, i) => detector.process(level, i * FRAME_MS))
    .filter(Boolean);
}

test("two quick claps are a double clap", () => {
  assert.deepEqual(events(signal(2000, [[500, CLAP], [800, CLAP]])), ["clap", "double"]);
});

test("claps too far apart are not a double clap", () => {
  assert.deepEqual(events(signal(3000, [[500, CLAP], [1600, CLAP]])), ["clap", "clap"]);
});

test("sustained sound like speech is not a clap", () => {
  assert.deepEqual(events(signal(3000, [[500, SPEECH], [1000, SPEECH]])), []);
});

test("quiet taps below the threshold are ignored", () => {
  const tap = [0.15, 0.05, 0.01];
  assert.deepEqual(events(signal(2000, [[500, tap], [800, tap]])), []);
});

test("cooldown stops a burst of claps from waking twice", () => {
  const claps = [500, 800, 1100, 1400].map((t) => [t, CLAP]);
  assert.deepEqual(events(signal(3000, claps)).filter((e) => e === "double"), ["double"]);
});

test("loud background raises the bar", () => {
  const noisy = signal(3000, [[1500, CLAP], [1800, CLAP]]).map((l, i) => (i < 150 ? 0.12 : l));
  // floor adapts to ~0.12, so the threshold becomes ~0.7: the 0.85 claps still count
  assert.ok(events(noisy).includes("double"));
});

test("wake word with a command after it", () => {
  assert.deepEqual(matchWakeWord("Hey Nova, what's on my calendar?", "Nova"), {
    command: "what's on my calendar?",
  });
});

test("wake word alone", () => {
  assert.deepEqual(matchWakeWord("nova", "Nova"), { command: "" });
});

test("slightly misheard long names still match", () => {
  assert.deepEqual(matchWakeWord("ok jarvas turn on focus mode", "Jarvis"), {
    command: "turn on focus mode",
  });
});

test("short names must match exactly, and other words don't wake it", () => {
  assert.equal(matchWakeWord("nina what time is it", "Nova"), null);
  assert.equal(matchWakeWord("what time is it", "Assistant"), null);
});

test("multi-word wake phrase", () => {
  assert.deepEqual(matchWakeWord("hey home base, lights off", "Home Base"), {
    command: "lights off",
  });
});

test("speech chunks are short and skip markdown and code", () => {
  const text = "**Done.** Here is code:\n```py\nprint(1)\n```\n" + "Sentence. ".repeat(40);
  const chunks = speechChunks(text);
  assert.ok(chunks.every((c) => c.length <= 230));
  assert.ok(!chunks.join(" ").includes("print(1)"));
  assert.ok(!chunks.join(" ").includes("*"));
});
