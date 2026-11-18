"""A test-only sandbox driver (M15): counts the lines of the machine file it is given. It runs only inside the
sandbox child, so it imports nothing from kaful. Payload: target_file (set by run_driver)."""

from pathlib import Path
from typing import Any


def count(payload: dict[str, Any]) -> dict[str, Any]:
    text = Path(payload["target_file"]).read_text(encoding="utf-8")
    return {"lines": len(text.splitlines())}
