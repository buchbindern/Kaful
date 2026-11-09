# CDV hold-out: understand run 20261109T143015Z-596dd5 (FAILED GATE)

The one M7b hold-out run on CDV, kept as the record (Natalie, 2026-11-09). **Not adjudicated.**

- **Gate.** The understand phase failed its unit gate (`ºC`, U+00BA, on spike_temp_2), so
  `kaful understand score` refuses the run. These files were made outside the CLI, by the script below,
  which calls the same `score()` on the run's three artifacts.
- **Golden labels.** Scored against `tests/fixtures/understand/golden/cdv.toml` at commit 5ff8e02
  (sha256 cf17340e3622…), as frozen before the run. Later edits change notes only.
- **Files.**
  - `…__failed-gate.json`: the raw score, with no aliases applied.
  - `…__failed-gate.md`: the disagreement table. Claude pre-filled every row (88 rows, 17 marked unsure);
    the verdict column is empty.
- **Hand-alias figures.** These come from that table. The first number counts the confident aliases; the
  number in brackets adds the unsure ones:
  - fault modes R 0.07 (0.14), P 0.20;
  - carriers 5 of 13;
  - channels R / P 0.35 / 0.50, roles 0.67 (0.71);
  - components R / P 0.48 / 0.53.

  docs/milestones/M7b.md has the full exit table and the causes.
- **Seen once.** M7c re-scores CDV once, labelled "seen once, not a clean hold-out".

```python
"""Measure a failed understand run against its golden labels; writes to the scratchpad only (not an output)."""
import sys
from pathlib import Path
from kaful.artifacts.machine_model import MachineModel
from kaful.artifacts.sensor_contract import SensorContract
from kaful.artifacts.understand_report import UnderstandReport
from kaful.config import KafulSettings
from kaful.phases.p2_understand.tools.commands import GOLDEN_DIR, _summary, disagreement_table
from kaful.phases.p2_understand.tools.golden_score import score
from kaful.runs import dump_json
machine, run, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
settings = KafulSettings()
d = settings.machine_dir(machine) / "outputs" / "runs" / run / "understand"
m = MachineModel.model_validate_json((d / "machine_model.json").read_text())
c = SensorContract.model_validate_json((d / "sensor_contract.json").read_text())
r = UnderstandReport.model_validate_json((d / "understand_report.json").read_text())
res = score(GOLDEN_DIR / f"{machine}.toml", m, c, r, settings.understand)
(out / f"{run}.json").write_text(dump_json(res)); (out / f"{run}.md").write_text(disagreement_table(res))
print(_summary(res))
```
