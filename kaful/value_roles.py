"""Value roles: what a quoted value is, and whether it may set a prior (M7 plan, M9 rule).

The LLM proposes a role for each value it extracts: spec, variant, default,
setting_range or example (kaful.artifacts.base.ValueRole). Code may only downgrade a
proposal to `example`, never upgrade one:
- a value found only in a screenshot or photo section of the page's LLM layer (ingest
  renders those as "Figure (screenshot)" / "Figure (photo)") is an example: a display
  shows one state of the machine, not its spec;
- a value whose quote follows an example marker ("example", "e.g.", ...) within
  `example_window_tokens` tokens in the page's raw text is from a worked example.
In M9 only spec, default and setting_range set priors; a variant only when machine.toml
declares which configuration applies; an example never (at most a labelled hint).
"""

import re
from difflib import SequenceMatcher

from kaful.artifacts.base import Evidence, ValueRole
from kaful.artifacts.document_index import PageText
from kaful.config import UnderstandSettings
from kaful.evidence import tokens

PRIOR_ROLES: frozenset[ValueRole] = frozenset({"spec", "default", "setting_range"})
FIGURE_HEADER = re.compile(r"^Figure \((?P<kind>[a-z_]+)\)")


def may_set_prior(role: ValueRole | None, configuration_declared: bool) -> bool:
    """Whether a value with this role may set an M9 prior (a variant needs the configuration declared)."""
    return role in PRIOR_ROLES or (role == "variant" and configuration_declared)


def _locate(quote: list[str], text: list[str], near_ratio: float) -> int | None:
    """Start of the quote's tokens in `text`: exact if possible, else the best window above `near_ratio`."""
    n = len(quote)
    if not n:
        return None
    for i in range(len(text) - n + 1):
        if text[i:i + n] == quote:
            return i
    best = max(range(max(1, len(text) - n + 1)), key=lambda i: SequenceMatcher(None, quote, text[i:i + n]).ratio(),
               default=None)
    if best is None or SequenceMatcher(None, quote, text[best:best + n]).ratio() < near_ratio:
        return None
    return best


def _after_marker(quote: str, raw: str, settings: UnderstandSettings) -> bool:
    text = tokens(raw)
    start = _locate(tokens(quote), text, settings.evidence_near_ratio)
    if start is None:
        return False
    before = text[max(0, start - settings.example_window_tokens):start]
    for marker in settings.example_markers:
        m = tokens(marker)
        if any(before[i:i + len(m)] == m for i in range(len(before) - len(m) + 1)):
            return True
    return False


def in_example_figure(quote: str, page: PageText, settings: UnderstandSettings) -> bool:
    q = tokens(quote)
    for layer in page.layers:
        if layer.kind != "llm":
            continue
        for section in layer.text.split("\n\n"):
            header = FIGURE_HEADER.match(section)
            if header and header["kind"] in settings.example_figure_kinds and \
                    _locate(q, tokens(section), settings.evidence_near_ratio) is not None:
                return True
    return False


def settle_role(proposed: ValueRole, evidence: Evidence, page: PageText | None,
                settings: UnderstandSettings) -> tuple[ValueRole, str | None]:
    """The role to record, and why code downgraded the proposal (None when it kept it)."""
    if proposed == "example" or page is None:
        return proposed, None
    if evidence.verification == "in_layer" and in_example_figure(evidence.quote, page, settings):
        return "example", "found only in a screenshot or photo"
    if evidence.verification in ("verbatim", "near") and _after_marker(evidence.quote, page.text, settings):
        return "example", "quoted from a worked example"
    return proposed, None
