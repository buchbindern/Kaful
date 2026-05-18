"""
utils/llm.py
------------
Clean wrapper around Claude API calls.

No RAG, no pipeline logic — just sends prompts and returns text.
If you want to inject RAG context, query it first then pass it in the prompt.

Usage:
    from utils.llm import call_claude

    response = call_claude("What is the brew temperature?")
    response = call_claude(prompt, model=CODEGEN_MODEL, max_tokens=4000)
"""

import os
import time
import random

import anthropic
from anthropic import RateLimitError


# Initialise client once at module level
_client = None

def _get_client() -> anthropic.Anthropic:
    """Get or create the Anthropic client."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client

def call_claude(
    prompt: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4000,
    system: str = None,
    temperature: float = 1.0,
    max_retries: int = 5,
) -> str:
    """
    Call Claude and return the response text.

    Automatically retries on rate limit (429) and overload (529) errors
    with exponential backoff.

    Args:
        prompt:      the user message to send
        model:       Claude model to use
        max_tokens:  maximum tokens in the response
        system:      optional system prompt
        max_retries: number of retry attempts on transient errors

    Returns:
        response text as a string

    Raises:
        anthropic.APIError: on non-retryable errors
        Exception: if max retries exceeded
    """
    client   = _get_client()
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model":       model,
                "max_tokens":  max_tokens,
                "messages":    messages,
                "temperature": temperature,
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)
            return response.content[0].text

        except RateLimitError:
            _handle_retry(attempt, max_retries, reason="Rate limit (429)")

        except anthropic.APIStatusError as e:
            if e.status_code == 529:
                _handle_retry(attempt, max_retries, reason="Overloaded (529)")
            else:
                raise  # non-retryable — surface immediately

        except Exception:
            raise

    raise Exception(f"Claude call failed after {max_retries} attempts.")


def _handle_retry(attempt: int, max_retries: int, reason: str):
    """Wait with exponential backoff before retrying."""
    if attempt < max_retries - 1:
        wait = (2 ** attempt) + random.uniform(0, 1)
        print(f"\n  {reason} — retrying in {wait:.1f}s ({attempt+2}/{max_retries})...")
        time.sleep(wait)
    else:
        raise Exception(f"Failed after {max_retries} attempts: {reason}")