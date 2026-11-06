"""Exact token counts for embedding models, offline.

tiktoken downloads its encoding files on first use; here the file is committed under
`kaful/llm/encodings/` and loaded from disk after its sha256 is checked, so token
counting never touches the network (tests included). Only the encodings listed in
ENCODINGS are available; an unknown one is an error.
"""

import base64
import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import tiktoken

ENCODINGS_DIR = Path(__file__).resolve().parent / "encodings"


@dataclass(frozen=True)
class EncodingSpec:
    file: str
    sha256: str
    pat_str: str
    special_tokens: dict[str, int]


# Copied from tiktoken_ext.openai_public.cl100k_base (tiktoken 0.13.0), including the sha256
# tiktoken itself expects for the file.
ENCODINGS: dict[str, EncodingSpec] = {
    "cl100k_base": EncodingSpec(
        file="cl100k_base.tiktoken",
        sha256="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        pat_str=r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
        special_tokens={"<|endoftext|>": 100257, "<|fim_prefix|>": 100258, "<|fim_middle|>": 100259,
                        "<|fim_suffix|>": 100260, "<|endofprompt|>": 100276},
    ),
}


class EncodingError(RuntimeError):
    """The encoding is unknown, or its committed file is missing or altered."""


@cache
def encoding(name: str, directory: Path = ENCODINGS_DIR) -> tiktoken.Encoding:
    spec = ENCODINGS.get(name)
    if spec is None:
        raise EncodingError(f"unknown encoding {name!r}; known: {sorted(ENCODINGS)}")
    path = directory / spec.file
    if not path.is_file():
        raise EncodingError(f"{path} is missing: the {name} encoding file must be committed (no download)")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != spec.sha256:
        raise EncodingError(f"{path}: sha256 does not match the expected {name} file")
    ranks = {base64.b64decode(token): int(rank) for token, rank in (line.split() for line in data.splitlines() if line)}
    return tiktoken.Encoding(name, pat_str=spec.pat_str, mergeable_ranks=ranks, special_tokens=spec.special_tokens)


def count(text: str, name: str) -> int:
    """Tokens in `text`; special-token strings count as plain text."""
    return len(encoding(name).encode(text, disallowed_special=()))


def window_spans(text: str, name: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Byte spans [start, end) of `text` (UTF-8) cut into windows of at most `size` tokens, each
    starting `overlap` tokens before the previous one ended. Cuts fall on token boundaries
    that start a character, so every span decodes. One span if the text fits."""
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap ({overlap}) < size ({size})")
    enc = encoding(name)
    ids = enc.encode(text, disallowed_special=())
    raw = text.encode("utf-8")
    if len(ids) <= size:
        return [(0, len(raw))]
    offsets = [0]
    for token in ids:
        offsets.append(offsets[-1] + len(enc.decode_single_token_bytes(token)))

    def at_char(i: int) -> int:  # the nearest token boundary at or before i that starts a character
        while 0 < i < len(ids) and raw[offsets[i]] & 0xC0 == 0x80:
            i -= 1
        return i

    spans, start = [], 0
    while True:
        end = len(ids) if start + size >= len(ids) else at_char(start + size)
        if end <= start:
            raise ValueError("a window would be empty (size too small for one character)")
        spans.append((offsets[start], offsets[end]))
        if end == len(ids):
            return spans
        start = at_char(max(end - overlap, start + 1))


def windows(text: str, name: str, size: int, overlap: int) -> list[str]:
    """The texts of `window_spans`."""
    raw = text.encode("utf-8")
    return [raw[a:b].decode("utf-8") for a, b in window_spans(text, name, size, overlap)]
