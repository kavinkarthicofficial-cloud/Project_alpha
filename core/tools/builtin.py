from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from core.llm.types import ToolSpec
from core.tools.registry import PermissionLevel, Tool, ToolRegistry, dumps


async def get_current_time(args: dict[str, Any]) -> str:
    tz_name = args.get("timezone")
    try:
        tz = ZoneInfo(tz_name) if tz_name else None
    except ZoneInfoNotFoundError:
        return dumps({"error": f"unknown timezone {tz_name!r}"})
    now = datetime.now(tz).astimezone(tz)
    return dumps({"iso": now.isoformat(timespec="seconds"), "weekday": now.strftime("%A")})


GET_CURRENT_TIME = Tool(
    spec=ToolSpec(
        name="get_current_time",
        description=(
            "Get the current date, time and weekday. Call this whenever the answer depends on "
            "today's date or the current time, such as deadlines, ages or 'how long until'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA timezone such as 'Asia/Kolkata'. Omit for the server's local time.",
                }
            },
            "additionalProperties": False,
        },
    ),
    level=PermissionLevel.READ,
    handler=get_current_time,
)


def default_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GET_CURRENT_TIME)
    return registry
