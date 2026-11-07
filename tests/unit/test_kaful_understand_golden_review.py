"""The understand golden-label review page: one card per row, page images with printed labels,
and the original file embedded so the export changes only the answers."""

import json
import re
from pathlib import Path

from kaful.config import IngestSettings
from kaful.phases.p2_understand.tools.golden_review import review_html
from tests.harness.kaful_ingest import make_pdf_machine

GOLDEN = """schema_version = 1
machine_id = "pdf_machine"
status = "draft"

# 2. Candidate channels.
[[candidate_channels]]
name = "flow_rate"
suggested_role = "operating_input"
unit = "m3/h"
doc_id = "tp_manual"
page = 2
quote = "Rated flow"
why = "listed on page 2"
verdict = ""
role = ""
note = ""

[[components]]
id = "pump"
legacy_category = "pump"
legacy_evidence = "the pump"
why = "main component"
verdict = "child"
parent = "skid"
note = ""
"""


def test_the_page_has_a_card_per_row_and_embeds_the_file(tmp_path: Path) -> None:
    machine_dir = make_pdf_machine(tmp_path)
    golden = tmp_path / "pdf_machine.toml"
    golden.write_text(GOLDEN)
    page = review_html(golden, machine_dir, IngestSettings(label_min_pages=3))
    assert 'id="candidate_channels/0"' in page and 'id="components/0"' in page
    assert "tp_manual p2 (label 2)" in page and "data:image/png;base64," in page
    meta = json.loads(re.search(r"const META = (\{.*?\});", page, re.S).group(1))  # type: ignore[union-attr]
    assert meta["text"] == GOLDEN and meta["rows"] == {"candidate_channels": [None], "components": [None]}
    assert meta["answers"]["components"] == ["verdict", "parent", "note"]
    assert meta["answers"]["candidate_fault_modes"] == ["verdict", "note"]


def test_answers_already_in_the_file_seed_the_page(tmp_path: Path) -> None:
    """Opened in a fresh browser, the page must show (and re-export) the answers the file has."""
    machine_dir = make_pdf_machine(tmp_path)
    golden = tmp_path / "pdf_machine.toml"
    golden.write_text(GOLDEN)
    page = review_html(golden, machine_dir, IngestSettings(label_min_pages=3))
    meta = json.loads(re.search(r"const META = (\{.*?\});", page, re.S).group(1))  # type: ignore[union-attr]
    assert meta["given"] == {"components/0": {"verdict": "child", "parent": "skid"}}


def test_the_page_records_secondary_roles_and_fault_mode_documents(tmp_path: Path) -> None:
    machine_dir = make_pdf_machine(tmp_path)
    golden = tmp_path / "pdf_machine.toml"
    golden.write_text(GOLDEN + """
[[fault_modes]]
component = "pump"
what_degrades = "seal"
verdict = "yes"
doc_id = "tp_manual"
page = "2"
page_label = "2"
note = ""
""")
    page = review_html(golden, machine_dir, IngestSettings(label_min_pages=3))
    meta = json.loads(re.search(r"const META = (\{.*?\});", page, re.S).group(1))  # type: ignore[union-attr]
    assert meta["answers"]["candidate_channels"] == ["verdict", "role", "secondary_role", "note"]
    assert meta["answers"]["fault_modes"] == ["verdict", "doc_id", "page", "note"]
    assert "tp_manual p2 (label 2)" in page.split('id="fault_modes/0"')[1]  # a page kept as text still shows


def test_browser_progress_is_keyed_to_the_file_it_was_made_for(tmp_path: Path) -> None:
    """After the file changes (an edit, a decision applied), old browser answers must not overwrite it."""
    machine_dir = make_pdf_machine(tmp_path)
    golden = tmp_path / "pdf_machine.toml"
    keys = []
    for text in (GOLDEN, GOLDEN.replace('parent = "skid"', 'parent = "frame"')):
        golden.write_text(text)
        page = review_html(golden, machine_dir, IngestSettings(label_min_pages=3))
        keys.append(json.loads(re.search(r"const META = (\{.*?\});", page, re.S).group(1))["key"])  # type: ignore[union-attr]
    assert keys[0] != keys[1]
