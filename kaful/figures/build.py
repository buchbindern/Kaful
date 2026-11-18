"""`kaful figures build`: draw every figure of the spec and write SVG, PNG, CSV and the figures manifest.

For each figure: open its lineages (recording every file read), extract the data, render it under the spec's
style, and write <id>_<slug>.svg/.png/.csv beside the spec. The manifest lists per figure its inputs (repo-relative
path, sha256), its outputs' sha256, how its unit was chosen, and the code version; `--only` rebuilds some figures
and keeps the others' entries.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from pydantic import Field

from kaful.artifacts.base import NonEmptyStr, Sha256, StrictModel
from kaful.artifacts.manifest import CodeVersion
from kaful.config import REPO_ROOT, KafulSettings
from kaful.figures import (
    f1_pipeline,
    f2_tracking,
    f3_failure_time,
    f4_calibration,
    f5_cost,
    f6_provenance,
    f7_sharpness,
    f8_maintenance,
    f9_whatif,
    f11_failure_grid,
)
from kaful.figures.drawn import Context, Draw, Drawn
from kaful.figures.sources import Inputs
from kaful.figures import report as report_module
from kaful.figures.spec import FigureBase, FigureId, FiguresSpec, load_spec
from kaful.figures.style import csv_bytes, encode, rc
from kaful.runs import code_version, sha256_bytes, sha256_file, write_text_atomic

MANIFEST = "figures_manifest.json"
DRAWERS: dict[str, Draw] = {
    "pipeline": f1_pipeline.draw, "tracking": f2_tracking.draw, "failure_time": f3_failure_time.draw,
    "calibration": f4_calibration.draw, "cost": f5_cost.draw, "provenance": f6_provenance.draw,
    "sharpness": f7_sharpness.draw, "maintenance": f8_maintenance.draw, "whatif": f9_whatif.draw,
    "failure_grid": f11_failure_grid.draw,
}


class InputRecord(StrictModel):
    path: NonEmptyStr = Field(description="Relative to the repo root.")
    sha256: Sha256


class FigureRecord(StrictModel):
    id: FigureId
    kind: NonEmptyStr
    caption: NonEmptyStr
    outputs: dict[str, Sha256] = Field(description="File name -> sha256.")
    inputs: tuple[InputRecord, ...]
    selection: str | None = None
    notes: tuple[NonEmptyStr, ...] = ()


class ReportRecord(StrictModel):
    file: NonEmptyStr
    sha256: Sha256
    inputs: tuple[InputRecord, ...] = Field(description="Artifacts read, and every embedded figure's SVG.")


class FiguresManifest(StrictModel):
    schema_version: int = 1
    kind: str = "figures_manifest"
    spec: InputRecord
    code: CodeVersion
    figures: tuple[FigureRecord, ...]
    report: ReportRecord | None = None


def _write(out_dir: Path, stem: str, drawn: Drawn, ctx: Context) -> dict[str, str]:
    files = {f"{stem}.{ext}": data for ext, data in encode(drawn.figure, ctx.style).items()}
    files[f"{stem}.csv"] = csv_bytes(drawn.columns, drawn.rows)
    out = {}
    for name, data in sorted(files.items()):
        (out_dir / name).write_bytes(data)
        out[name] = sha256_bytes(data)
    return out


def build_one(spec: FiguresSpec, figure_id: str, settings: KafulSettings, out_dir: Path,
              root: Path = REPO_ROOT) -> FigureRecord:
    f = next(x for x in spec.figures if x.id == figure_id)
    ctx = Context(spec=spec, settings=settings, inputs=Inputs(root=root))
    with plt.rc_context(rc(spec.style)):
        drawn = DRAWERS[f.kind](f, ctx)
        outputs = _write(out_dir, spec.stem(f), drawn, ctx)
    return FigureRecord(id=f.id, kind=f.kind, caption=f.caption, outputs=outputs,
                        inputs=tuple(InputRecord(path=p, sha256=s) for p, s in ctx.inputs.sorted()),
                        selection=drawn.selection, notes=drawn.notes)


def build(spec_path: Path, settings: KafulSettings, only: tuple[str, ...] = (),
          root: Path = REPO_ROOT) -> FiguresManifest:
    """`root`: what input paths are recorded relative to (the repo; a test's folder in tests). The code version is
    read before anything is written, so the build's own outputs never count as a dirty checkout."""
    code = code_version()
    spec = load_spec(spec_path)
    unknown = sorted(set(only) - {f.id for f in spec.figures})
    if unknown:
        raise ValueError(f"unknown figure ids: {unknown}")
    out_dir = spec_path.parent
    kept: dict[str, FigureRecord] = {}
    if only and (out_dir / MANIFEST).is_file():
        old = FiguresManifest.model_validate_json((out_dir / MANIFEST).read_text(encoding="utf-8"))
        kept = {r.id: r for r in old.figures if r.id not in only}
    built = {f.id: build_one(spec, f.id, settings, out_dir, root) for f in spec.figures if not only or f.id in only}
    records = {**kept, **built}
    figures = tuple(records[f.id] for f in spec.figures if f.id in records)
    report = build_report_file(spec, figures, settings, out_dir, root, code) if spec.report else None
    manifest = FiguresManifest(
        spec=InputRecord(path=_rel(spec_path, root), sha256=sha256_file(spec_path)), code=code,
        figures=figures, report=report)
    write_text_atomic(out_dir / MANIFEST, manifest.model_dump_json(indent=2) + "\n")
    return manifest


def build_report_file(spec: FiguresSpec, records: tuple[FigureRecord, ...], settings: KafulSettings, out_dir: Path,
                      root: Path, code: CodeVersion) -> ReportRecord:
    """The demo report from the figures as recorded: each SVG is checked against its manifest sha256."""
    assert spec.report is not None
    ctx = Context(spec=spec, settings=settings, inputs=Inputs(root=root))
    by_id = {r.id: r for r in records}
    svgs: dict[str, tuple[FigureBase, bytes]] = {}
    for f in spec.figures:
        name = f"{spec.stem(f)}.svg"
        if f.id not in by_id or name not in by_id[f.id].outputs:
            raise ValueError(f"report: {f.id} has not been built")
        data = ctx.inputs.add(out_dir / name).read_bytes()
        if sha256_bytes(data) != by_id[f.id].outputs[name]:
            raise ValueError(f"report: {name} does not match its manifest sha256; rebuild it")
        svgs[f.id] = (f, data)
    with plt.rc_context(rc(spec.style)):
        text = report_module.build_report(spec.report, ctx, svgs, code.git_commit)
    target = out_dir / spec.report.file
    write_text_atomic(target, text)
    return ReportRecord(file=spec.report.file, sha256=sha256_file(target),
                        inputs=tuple(InputRecord(path=p, sha256=s) for p, s in ctx.inputs.sorted()))


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
