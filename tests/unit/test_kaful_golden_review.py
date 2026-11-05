"""Golden review page: one judgeable item per expected value; pages render or are flagged missing."""

from pathlib import Path

from kaful.config import KafulSettings
from kaful.evals.golden_review import expected_items, write_review
from kaful.evals.models import EvalCase, EvalSet
from kaful.prompts.registry import PromptRegistry


def test_expected_items_split_lists_keys_and_scalars() -> None:
    case = EvalCase.model_validate({
        "id": "c", "description": "d", "inputs": {"doc": "x"},
        "golden": [{"metric": "m", "field": "f", "compare": "contains_all", "expected": ["a | 1", "b | 2"]},
                   {"metric": "k", "field": "g", "compare": "keyed", "key": "n", "value": "v",
                    "expected": {"z": 1, "y": "2"}},
                   {"metric": "e", "field": "h", "compare": "exact", "expected": 3}]})
    assert [(i.golden, i.item, i.text) for i in expected_items(case)] == [
        (0, 0, "a | 1"), (0, 1, "b | 2"), (1, 0, "y → 2"), (1, 1, "z → 1"), (2, 0, "3")]


def test_review_page_lists_every_golden_case_and_flags_missing_pages(tmp_path: Path) -> None:
    prompt = PromptRegistry().load("ingest.extract_page")
    evalset = EvalSet(prompt, KafulSettings(machines_dir=tmp_path / "none"))
    path = write_review(tmp_path, evalset)
    text = path.read_text()
    assert path.name == f"ingest.extract_page__v{prompt.version}.html"
    assert all(f'id="{c.id}"' in text for c in evalset.cases if "golden" in c.tags)
    assert "Source page not on disk" in text and "golden_review.toml" in text
