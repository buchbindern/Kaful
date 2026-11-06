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
verdict = ""
parent = ""
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
