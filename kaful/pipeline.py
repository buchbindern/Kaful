"""Start a run and execute the selected phases in order.

Phases before the first selected one are reused from a base run (see
`runs.select_base_run`). Phases without an implementation are recorded as
`not_implemented`; nothing is silently skipped.
"""

import json
import logging
import os
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from kaful.artifacts.manifest import BaseRun, Override, PhaseRecord, RunManifest, RunStatus
from kaful.config import KafulSettings, load_machine_config
from kaful.llm.batch import BatchPending
from kaful.llm.budget import BudgetExceeded, RunBudget
from kaful.llm.cache import ResponseStore
from kaful.llm.gateway import Gateway
from kaful.llm.ledger import LEDGER_FILE, Ledger, cost_by_phase
from kaful.llm.transport import Transport
from kaful.logs import configure_logging, log_context
from kaful.phases import (
    OPTIONAL_PHASES,
    PHASE_IMPLEMENTATIONS,
    PHASE_ORDER,
    PhaseName,
    select_phases,
    upstream_phases,
)
from kaful.runs import (
    LLM_BATCHES_SUBDIR,
    LLM_CACHE_SUBDIR,
    LOG_FILE,
    REUSABLE_STATUSES,
    RunContext,
    code_version,
    machine_inputs,
    new_run_id,
    now_utc,
    runs_dir,
    select_base_run,
    sha256_bytes,
)

log = logging.getLogger("kaful.pipeline")


@dataclass(frozen=True)
class RunRequest:
    """What `kaful run` was asked to do."""

    machine_id: str
    argv: tuple[str, ...]
    from_phase: PhaseName | None = None
    only: tuple[PhaseName, ...] = ()
    budget_usd: float | None = None
    allow_failures: bool = False
    base_run: str | None = None


def start_run(
    settings: KafulSettings, request: RunRequest, index_hash: str | None = None,
    transport: Transport | None = None,
) -> tuple[RunContext, RunManifest | None]:
    """Validate inputs, choose the base run, create the run directory and manifest.

    `index_hash` is the hash of the current document index; ingest (M6) supplies it.
    The run gets an LLM gateway over `transport` (None: answers only from the cache).
    """
    machine_dir = settings.machine_dir(request.machine_id)
    machine = load_machine_config(machine_dir)
    selected = select_phases(request.from_phase, request.only)
    inputs = machine_inputs(machine_dir, machine)
    needed = upstream_phases(selected)
    started = now_utc()
    base: RunManifest | None = None
    base_run: BaseRun | None = None
    diffs: list[str] = []
    if needed:
        base, base_run, diffs = select_base_run(
            machine_dir, needed, inputs, index_hash, request.base_run
        )
    overrides = _overrides(request, base_run, diffs, started)
    snapshot = settings.snapshot()
    run_id = new_run_id(started)
    manifest = RunManifest(
        run_id=run_id,
        machine_id=machine.machine_id,
        created_at=started,
        code=code_version(),
        argv=request.argv,
        from_phase=request.from_phase,
        only=request.only,
        base_run=base_run,
        budget_usd=settings.default_budget_usd if request.budget_usd is None else request.budget_usd,
        allow_failures=request.allow_failures,
        config=snapshot,
        config_hash=sha256_bytes(json.dumps(snapshot, sort_keys=True).encode()),
        inputs=inputs,
        index_hash=base.index_hash if base is not None else index_hash,
        python=platform.python_version(),
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        overrides=tuple(overrides),
        phases=tuple(PhaseRecord(phase=p) for p in PHASE_ORDER),
    )
    run_dir = runs_dir(machine_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ctx = RunContext(
        settings=settings, machine=machine, machine_dir=machine_dir, run_dir=run_dir, manifest=manifest
    )
    ctx.gateway = make_gateway(ctx, transport)
    ctx.write_manifest()
    return ctx, base


def make_gateway(ctx: RunContext, transport: Transport | None) -> Gateway:
    """The run's gateway: ledger in the run dir, response cache and batch state per machine."""
    return Gateway(
        settings=ctx.settings.llm,
        ledger=Ledger(ctx.run_dir / LEDGER_FILE),
        budget=RunBudget(ctx.manifest.budget_usd),
        transport=transport,
        cache=ResponseStore(ctx.machine_dir / LLM_CACHE_SUBDIR),
        batch_state_dir=ctx.machine_dir / LLM_BATCHES_SUBDIR,
        run_id=ctx.manifest.run_id,
        machine_id=ctx.machine.machine_id,
    )


def _overrides(
    request: RunRequest, base_run: BaseRun | None, diffs: list[str], at: datetime
) -> list[Override]:
    """Overrides to record: a base run with different inputs, and --allow-failures."""
    overrides = []
    if base_run is not None and diffs:
        overrides.append(Override(
            kind="base_run", target=base_run.run_id, value="inputs differ",
            reason="; ".join(diffs), recorded_at=at,
        ))
    if request.allow_failures:
        overrides.append(Override(
            kind="allow_failures", target="run", value="true",
            reason="--allow-failures given on the command line", recorded_at=at,
        ))
    return overrides


def execute(ctx: RunContext, selected: Sequence[PhaseName], base: RunManifest | None) -> RunManifest:
    """Run the selected phases in order, reusing upstream ones from `base`."""
    configure_logging(ctx.settings.log_level, ctx.run_dir / LOG_FILE)
    first = PHASE_ORDER.index(selected[0])
    with log_context(run_id=ctx.manifest.run_id):
        for index, phase in enumerate(PHASE_ORDER):
            if index < first:
                ctx.set_phase(_reused_or_skipped(phase, base))
            elif phase in selected:
                ctx.set_phase(_with_llm_accounting(ctx, _run_phase(ctx, phase)))
            else:
                ctx.set_phase(PhaseRecord(phase=phase, status="skipped"))
            phase_status = ctx.manifest.phase(phase).status
            if phase_status == "budget_exceeded" or (phase_status == "failed" and not ctx.manifest.allow_failures):
                break
        status = _run_status(ctx.manifest)
        ctx.manifest = ctx.manifest.model_copy(update={"status": status, "finished_at": now_utc()})
        ctx.write_manifest()
        log.info("run finished", extra={"fields": {"status": status}})
    return ctx.manifest


def _reused_or_skipped(phase: PhaseName, base: RunManifest | None) -> PhaseRecord:
    if base is None:
        return PhaseRecord(phase=phase, status="skipped")
    record = base.phase(phase)
    if record.status not in REUSABLE_STATUSES:
        if phase in OPTIONAL_PHASES:
            return PhaseRecord(phase=phase, status="skipped")
        raise AssertionError(f"base run {base.run_id} did not pass {phase}")
    return record.model_copy(update={"status": "reused"})


def _run_phase(ctx: RunContext, phase: PhaseName) -> PhaseRecord:
    with log_context(phase=phase):
        implementation = PHASE_IMPLEMENTATIONS.get(phase)
        if implementation is None:
            log.warning("phase not implemented")
            return PhaseRecord(phase=phase, status="not_implemented")
        try:
            return implementation.run(ctx)
        except BudgetExceeded as exc:
            log.error("budget exceeded; stopping the run", extra={"fields": {"error": str(exc)}})
            return PhaseRecord(phase=phase, status="budget_exceeded", failures=(str(exc),))
        except BatchPending as exc:
            log.error("batch still running; rerun to collect it", extra={"fields": {"batch_id": exc.batch_id}})
            return PhaseRecord(phase=phase, status="failed", failures=(str(exc),))


def _with_llm_accounting(ctx: RunContext, record: PhaseRecord) -> PhaseRecord:
    """Phase cost from the run ledger, and the prompt versions used so far, into the manifest."""
    if ctx.gateway is None:
        return record
    versions = {k: str(v) for k, v in sorted(ctx.gateway.prompt_versions.items())}
    ctx.manifest = ctx.manifest.model_copy(update={"prompt_versions": versions})
    cost = cost_by_phase(ctx.gateway.ledger.rows()).get(record.phase, 0.0)
    return record.model_copy(update={"cost_usd": cost})


def _run_status(manifest: RunManifest) -> RunStatus:
    statuses = {r.status for r in manifest.phases}
    if "budget_exceeded" in statuses:
        return "budget_exceeded"
    if "failed" in statuses:
        return "failed"
    if "not_implemented" in statuses:
        return "not_implemented"
    return "passed"

