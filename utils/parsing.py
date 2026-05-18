"""
utils/parsing.py
----------------
Helpers for parsing LLM responses.

Claude often wraps JSON in markdown fences or adds explanation text.
These functions clean that up so phases can work with structured data.

Usage:
    from utils.parsing import parse_json, strip_fences, is_valid_python
"""

import json
import re
import ast


def strip_fences(text: str) -> str:
    """
    Remove markdown code fences from LLM output.

    Handles:
        ```json ... ```
        ```python ... ```
        ``` ... ```

    Args:
        text: raw LLM response

    Returns:
        cleaned string with fences removed
    """
    if not text:
        return ""

    # Remove opening fence (with optional language tag) and closing fence
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())

    return text.strip()


def parse_json(text: str) -> dict | list | None:
    """
    Parse JSON from an LLM response, handling common issues.

    Tries in order:
    1. Direct parse
    2. Strip fences then parse
    3. Extract first JSON object/array found in the text

    Args:
        text: raw LLM response that should contain JSON

    Returns:
        parsed dict or list, or None if parsing fails
    """
    if not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try stripping fences
    try:
        return json.loads(strip_fences(text))
    except json.JSONDecodeError:
        pass

    # Try extracting first JSON object or array from text
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    return None


def is_valid_python(code: str) -> bool:
    """
    Check if a string is valid Python syntax.

    Args:
        code: Python code string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
