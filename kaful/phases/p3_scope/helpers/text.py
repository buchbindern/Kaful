"""Text units and component names for the life signal (b_find_life_statements) and named channels (a_collect_signals).

A page is read as sentences of its raw text (line breaks joined) and as rows of its table layers (markdown
tables: the ruled-table layer and the LLM layer), because maintenance schedules are tables whose raw text puts
every cell on its own line (air_compressor p29: the running hours sit under a "Running hours" header, and a
ditto mark repeats the period of the row above).

A unit names a component when its words contain one of the component's names as consecutive words: the
printed name, the id, an alias, each also without a trailing generic head word (module, unit, system) and
the printed name without its designation; words are compared lower case with a plural "s" dropped. A unit
also names it when it prints the component's designation in parentheses ("(AF)" for "Air filter (AF)") and
another word of its name: schedules name parts by designation (air_compressor p29 "Have oil separator (OS)
replaced" for "Oil separator element (OS)").

A channel's name or description names a component as the thing it measures when one of its name forms is
followed by a quantity word (settings channel_quantity_words), by punctuation or by nothing: "Coffee boiler
temperature" names the coffee boiler; "Flow rate of the coffee boiler flowmeter" does not (the flowmeter is
measured).
"""

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from kaful.artifacts.machine_model import Component

WORD_RE = re.compile(r"[a-z0-9]+")
TOKEN_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")
SENTENCE_END_RE = re.compile(r"(?<=[.!?;])\s+")
SPACE_RE = re.compile(r"\s+")
DITTO = {"“", "”", '"', "„", "″", "〃"}
EMPTY_CELL = set("—–-: ")  # a cell of dashes is an empty cell
TABLE_LAYER_KINDS = ("llm", "table_md")
DESIGNATION_RE = re.compile(r"\(([^()]{1,16})\)\s*$")
SHORTEST_PLURAL = 4  # a plural "s" is dropped only from words this long (bearings -> bearing, gas stays)


def stem(word: str) -> str:
    return word[:-1] if len(word) >= SHORTEST_PLURAL and word.endswith("s") else word


def words(text: str) -> list[str]:
    return [stem(w) for w in WORD_RE.findall(text.lower().replace("-", " "))]


def has_phrase(haystack: Sequence[str], phrase: Sequence[str]) -> bool:
    n = len(phrase)
    return bool(n) and any(list(haystack[i:i + n]) == list(phrase) for i in range(len(haystack) - n + 1))


def designation(component: Component) -> str | None:
    """The designation printed in parentheses at the end of the name ("AF" in "Air filter (AF)")."""
    m = DESIGNATION_RE.search(component.name)
    return m.group(1).strip() if m else None


def names(component: Component, heads: Sequence[str]) -> list[list[str]]:
    """The word sequences a text may name the component by, longest first, without duplicates."""
    stems = {stem(h) for h in heads}
    found: list[list[str]] = []
    bare = DESIGNATION_RE.sub("", component.name)
    for text in (component.name, bare, component.id.replace("_", " "), *(a.replace("_", " ") for a in component.aliases)):
        w = words(text)
        for form in (w, w[:-1] if len(w) > 1 and w[-1] in stems else w):
            if form and form not in found:
                found.append(form)
    return sorted(found, key=lambda f: (-len(f), f))


def named(text: str, component: Component, forms: Sequence[Sequence[str]]) -> bool:
    """Whether `text` names the component: one of its name `forms` as consecutive words, or its designation in
    parentheses with another word of its name."""
    w = words(text)
    if any(has_phrase(w, f) for f in forms):
        return True
    mark = designation(component)
    return (mark is not None and f"({mark})" in text
            and bool(set(words(DESIGNATION_RE.sub("", component.name))) & set(w)))


def measures(text: str, forms: Sequence[Sequence[str]], quantities: Collection[str]) -> bool:
    """Whether `text` names one of the name `forms` as the thing measured (see the module docstring)."""
    tokens = [stem(t) if t[0].isalnum() else t for t in TOKEN_RE.findall(text.lower().replace("-", " "))]
    for form in forms:
        n = len(form)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] != list(form):
                continue
            after = tokens[i + n] if i + n < len(tokens) else ""
            if not after or not after[0].isalnum() or after in quantities:
                return True
    return False


def sentences(text: str) -> list[str]:
    """Raw text with line breaks joined, split after . ! ? ;"""
    return [s for s in SENTENCE_END_RE.split(SPACE_RE.sub(" ", text).strip()) if s]


@dataclass(frozen=True)
class Row:
    """One table row: its cells, the table's header cells, and the period it repeats by ditto (if any)."""

    cells: tuple[str, ...]
    header: tuple[str, ...]
    inherited: str = ""

    @property
    def text(self) -> str:
        return " ".join(c for c in self.cells if c and c not in DITTO and set(c) - EMPTY_CELL)


def _cells(line: str) -> tuple[str, ...]:
    return tuple(c.strip() for c in line.strip().strip("|").split("|"))


def table_rows(layer_text: str) -> list[Row]:
    """Rows of the markdown tables in a layer; a first cell that is a ditto mark inherits the last first cell."""
    rows: list[Row] = []
    header: tuple[str, ...] = ()
    last_first = ""
    previous_was_table = False
    for line in layer_text.splitlines():
        if not line.strip().startswith("|"):
            previous_was_table = False
            continue
        cells = _cells(line)
        if not previous_was_table:
            header, last_first, previous_was_table = cells, "", True
            continue
        if all(set(c) <= EMPTY_CELL - {"—", "–"} for c in cells):
            continue  # the separator row
        first = cells[0] if cells else ""
        inherited = last_first if first in DITTO else ""
        if first and first not in DITTO and set(first) - EMPTY_CELL:
            last_first = first
        rows.append(Row(cells, header, inherited))
    return rows
