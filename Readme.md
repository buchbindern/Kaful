# Kaful

Kaful turns equipment documentation (manuals, catalogues, standards) into a physics-based prognostic digital twin,
and forecasts each component's remaining useful life (RUL) as a probability distribution.

Every number in a twin carries its provenance: the manual page whose raw text contains its quote, a standard, a
calibration from a documented life or service interval, or an LLM judgement (deliberately wide). Every LLM output is
validated and evaluated, every call is logged with its cost, and every run records what produced it.

## The pipeline

| # | Phase | What it does | Artifact |
|---|---|---|---|
| 1 | ingest | page text, tables and labels; LLM transcription only for pages that need it | `DocumentIndex` |
| 2 | understand | components, flow paths, fault modes, and the sensor contract (what the machine reports) | `MachineModel`, `SensorContract` |
| 3 | scope | which components get full models | `Scope` |
| 4 | parameterize | every rate, threshold and nominal input as a prior with provenance | `ParamSet` |
| 5 | build | a degradation template per component, bound to its parameters | `TwinDefinition` |
| 6 | verify | static and dynamic checks; a failing component is quarantined or the run stops | `VerificationReport` |
| 7 | synthesize | optional, no-data case: observations plus hidden truth from the priors | `SyntheticDataset` |
| 8 | estimate | particle filter (also UKF, open-loop) and RUL forecasts, one observation at a time | `ForecastSet` |
| 9 | evaluate | grounding, prompt quality, twin quality, calibration, sensitivity, cost | `EvaluationReport` |

Without failure data, absolute degradation rates can be checked for consistency with the documentation and
physics, not for accuracy. On synthetic data the forecasts are **calibrated but not sharp**: the particle filter's
90% intervals hold the truth 88-100% of the time, but the intervals are wide, because most lives are judged rather
than documented. See `docs/milestones/M14.md` and `M15.md`.

## Setup

Python 3.11 (ProgPy 1.7.1 does not support 3.13+).

```bash
conda create -n pipeline311 python=3.11
conda activate pipeline311
pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt
cp .env.example .env          # add ANTHROPIC_API_KEY and OPENAI_API_KEY
```

`requirements*.txt` pin the direct dependencies; `constraints.txt` pins everything they pull in.

## Running

A machine is a folder `machines/<id>/` with a `machine.toml` (its documents, time unit, declared maintenance and
instrumentation) and its documents under `manual/`. Runs, caches and reports go to `machines/<id>/outputs/`.

```bash
python -B -m kaful run eversys_coffee_machine --budget 1.0          # the whole pipeline, stopping at $1.00
python -B -m kaful run eversys_coffee_machine --from estimate       # reuse a matching earlier run up to estimate
python -B -m kaful run eversys_coffee_machine --offline             # caches only: no API client, $0
python -B -m kaful evaluate report eversys_coffee_machine --run <run_id>
python -B -m kaful runs prune eversys_coffee_machine                # what can be deleted, and why (--apply)
python -B -m kaful --help
```

Every LLM call goes through one gateway: it caches responses by content hash, uses prompt caching and the Batch API,
logs each call's tokens and cost to the run's ledger, and stops the run cleanly at its budget.

## Docker

```bash
docker build --target runtime -t kaful .
docker run --rm --env-file .env -v "$PWD/machines:/app/machines" kaful run eversys_coffee_machine --budget 1.0
```

The `test` target adds the dev dependencies and runs the offline suite: `docker build --target test -t kaful-test .`
then `docker run --rm --network none kaful-test`.

## Tests

```bash
pytest -q             # unit and offline integration tests; the network is blocked
pytest -m evals       # prompt evals on recorded outputs
ruff check kaful tests scripts
mypy                  # kaful/ strict, tests/ standard
```

`pytest -m live` makes real API calls and is never run without approval. CI (GitHub Actions) runs ruff, mypy and
`pytest -q` inside the test image with no network.

## Layout

```
kaful/            the package: cli, config, artifacts, llm gateway, prompts, phases/p1_ingest .. p9_evaluate,
                  retrieval, templates, backends/progpy, data, estimation, scoring, sandbox, evals
machines/<id>/    machine.toml (committed); manual/, outputs/ (local)
evals/            recorded eval outputs and results per prompt version
docs/             milestone reports, the M1 cost baseline and the legacy verification baseline
tests/            unit, integration, evals, fixtures
```

`PLAN.md` holds the milestones and `OVERVIEW.md` the audit of the original prototype. The prototype itself was
removed in M15; it is at the git tags `pre-revamp` and `legacy-final`.
