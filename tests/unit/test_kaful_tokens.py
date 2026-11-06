"""Exact token counts and embedding windows (kaful.llm.tokens), offline."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from kaful.artifacts.document_index import VectorsRef
from kaful.llm import tokens
from kaful.llm.tokens import ENCODINGS_DIR, EncodingError, count, encoding, window_spans, windows

ENC = "cl100k_base"


def test_the_committed_encoding_loads_offline_and_counts_exactly() -> None:
    assert count("Relax Time; 2,0; [s]", ENC) == 12
    assert count("<|endoftext|>", ENC) > 1  # special-token strings count as plain text


def test_a_text_that_fits_is_one_window() -> None:
    text = "Cake Thickness 14,0 mm"
    assert windows(text, ENC, 8000, 200) == [text]


def test_windows_cover_the_text_with_the_requested_overlap() -> None:
    text = " ".join(f"param{i} = {i * 7} min-1 |" for i in range(600))
    raw = text.encode()
    spans = window_spans(text, ENC, 100, 20)
    assert spans[0][0] == 0 and spans[-1][1] == len(raw)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:], strict=False):
        assert a0 < b0 < a1 < b1  # each window starts inside the previous one and goes further
        assert 15 <= count(raw[b0:a1].decode(), ENC) <= 25  # ~20 tokens of overlap
    parts = windows(text, ENC, 100, 20)
    assert all(count(p, ENC) <= 100 for p in parts)
    rebuilt = raw[: spans[0][1]] + b"".join(raw[prev[1]: span[1]] for prev, span in zip(spans, spans[1:], strict=False))
    assert rebuilt.decode() == text  # no byte is lost


def test_cuts_never_split_a_character() -> None:
    text = "温度センサー異常 éàü " * 300
    spans = window_spans(text, ENC, 37, 5)
    raw = text.encode()
    assert len(spans) > 3 and all(raw[a:b].decode() for a, b in spans)  # every span decodes


def test_bad_windows_and_encodings_are_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        windows("x", ENC, 10, 10)
    with pytest.raises(EncodingError, match="unknown encoding"):
        encoding("o200k_base")
    with pytest.raises(EncodingError, match="is missing"):
        encoding(ENC, tmp_path)
    altered = tmp_path / "altered"
    altered.mkdir()
    shutil.copy(ENCODINGS_DIR / tokens.ENCODINGS[ENC].file, altered)
    with (altered / tokens.ENCODINGS[ENC].file).open("ab") as f:
        f.write(b"\n")
    with pytest.raises(EncodingError, match="sha256"):
        encoding(ENC, altered)


def test_row_chunk_must_map_every_row_in_order() -> None:
    base = {"path": "v.npy", "sha256": "a" * 64, "model": "m", "dimensions": 2}
    assert VectorsRef(**base, n_rows=3).chunk_of_rows() == (0, 1, 2)
    assert VectorsRef(**base, n_rows=4, row_chunk=(0, 1, 1, 2)).chunk_of_rows() == (0, 1, 1, 2)
    for bad in ((0, 2, 2, 3), (1, 1, 2, 3), (0, 1, 2)):
        with pytest.raises(ValueError, match="row_chunk"):
            VectorsRef(**base, n_rows=4, row_chunk=bad)
    assert np.asarray(VectorsRef(**base, n_rows=0).chunk_of_rows()).size == 0
