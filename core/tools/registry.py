from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import jsonschema

from core.llm.types import ToolCallBlock, ToolResultBlock, ToolSpec


class PermissionLevel(IntEnum):
    READ = 0          # runs automatically
    REVERSIBLE = 1    # runs automatically, user is notified
    EXTERNAL = 2      # irreversible or outward-facing: always needs the user's confirmation


ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class Tool:
    spec: ToolSpec
    level: PermissionLevel
    handler: ToolHandler


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    async def execute(self, call: ToolCallBlock) -> ToolResultBlock:
        def error(text: str) -> ToolResultBlock:
            return ToolResultBlock(
                tool_call_id=call.id, tool_name=call.name, content=text, is_error=True
            )

        tool = self._tools.get(call.name)
        if tool is None:
            return error(f"Unknown tool {call.name!r}.")
        if "__invalid_json__" in call.arguments:
            return error("Tool input was not valid JSON. Please call the tool again.")
        # Models (especially small local ones, and any model under eager input
        # streaming) can produce arguments that parse but don't match the schema.
        try:
            jsonschema.validate(call.arguments, tool.spec.input_schema)
        except jsonschema.ValidationError as e:
            return error(f"Invalid input: {e.message}")
        if tool.level >= PermissionLevel.EXTERNAL:
            # Confirmation flow arrives with the integrations in Phase 4.
            return error("This action needs the user's confirmation, which is not supported yet.")
        try:
            content = await tool.handler(call.arguments)
        except Exception as e:  # noqa: BLE001 - a failing tool must not end the conversation
            return error(f"Tool failed: {type(e).__name__}: {e}")
        return ToolResultBlock(tool_call_id=call.id, tool_name=call.name, content=content)


def dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)
