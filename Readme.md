# Digital Twin Pipeline

A pipeline for building physics-based digital twins of industrial machines from technical documentation.

Given a machine manual PDF, the pipeline automatically:
1. Ingests and indexes the manual into a vector database
2. Extracts a telemetry event schema
3. Generates a realistic data simulator with degradation modeling
4. Builds a structured machine model (components, flow paths, physics)
5. Triages components for ProgPy modeling
6. Generates executable ProgPy component classes and a composite model
7. Runs state estimation (particle filter) and RUL prediction (Monte Carlo)

## Requirements

- Python 3.11 (ProgPy does not support 3.13+)
- Anthropic API key
- OpenAI API key (for embeddings)

## Setup

```bash
# Create environment
conda create -n pipeline311 python=3.11
conda activate pipeline311

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API keys to .env
```

## Usage

```bash
cd pipeline

# Run full pipeline for a machine
python run.py eversys_coffee_machine

# Start from a specific phase
python run.py eversys_coffee_machine --from phase3

# Run only one phase
python run.py eversys_coffee_machine --only phase2

# Force re-ingest the manual
python run.py eversys_coffee_machine --only phase1 --force
```

## Adding a New Machine

1. Create a folder under `pipeline/machines/<machine_id>/`
2. Add the manual PDF to `machines/<machine_id>/manual/`
3. Create `machines/<machine_id>/queries.py` with machine-specific queries:

```python
# Schema extraction queries
schema_queries = [...]
process_queries = [...]

# Simulation queries (topic-tagged tuples)
simulation_queries = [
    ("operating_context", "..."),
    ("degradation_maintenance_and_failures", "..."),
    ("process_and_ranges", "..."),
]

# Twin comprehension queries
twin_comprehension_queries = [...]

# Optional: exclude specific components from codegen
manual_exclusions = []
```

4. Run the pipeline:
```bash
python run.py <machine_id>
```

## Project Structure

```
pipeline/
├── run.py                      ← entry point
├── config.py                   ← paths, models, tuning parameters
├── state.py                    ← pipeline checkpoint state
│
├── phases/
│   ├── phase1_ingest.py        ← PDF → ChromaDB
│   ├── phase2_schema/          ← schema extraction (steps a-g)
│   ├── phase3_simulate/        ← data simulator (steps a-d)
│   ├── phase4_twin/            ← machine model (steps a-d)
│   ├── phase5_triage/          ← component triage (steps a-c)
│   ├── phase6_codegen/         ← ProgPy codegen (steps a-d)
│   └── phase7_estimate/        ← state estimation + RUL (steps a-c)
│
├── utils/
│   ├── llm.py                  ← Claude API wrapper
│   ├── rag.py                  ← ManualRAG (ingest + retrieve)
│   ├── progpy_rag.py           ← ProgPy framework context
│   ├── parsing.py              ← JSON parsing, code validation
│   └── helpers.py              ← chunk deduplication, context formatting
│
├── machines/
│   └── eversys_coffee_machine/
│       ├── manual/             ← PDF goes here (gitignored)
│       ├── rag/                ← ChromaDB vector store (gitignored)
│       ├── queries.py          ← machine-specific queries
│       └── outputs/            ← all pipeline outputs
│           ├── schema/         ← final_schema.json, DDL
│           ├── simulate/       ← simulator.py, events.csv
│           ├── twin/           ← machine model, component physics
│           ├── triage/         ← field assignments, triage results
│           ├── codegen/        ← specs, components, composite model
│           └── estimate/       ← particle filter results, RUL
│
└── rag_frameworks/
    └── progpy/                 ← ProgPy documentation RAG
```

## Pipeline Checkpoints

Every step saves its output to disk and skips if already done. To rerun a specific step, delete its output file and rerun the phase.

For example, to rerun schema merging (phase 2, step e):
```bash
rm machines/eversys_coffee_machine/outputs/schema/step_e_merged.json
rm machines/eversys_coffee_machine/outputs/schema/final_schema.json
python run.py eversys_coffee_machine --only phase2
```

## Models Used

| Task | Model |
|------|-------|
| Manual extraction | claude-sonnet-4-20250514 |
| Schema, planning, specs | claude-sonnet-4-20250514 |
| Component codegen | claude-sonnet-4-20250514 |
| Composite codegen | claude-opus-4-6 |
| Embeddings | text-embedding-3-large (OpenAI) |

## Known Limitations / Tweaks Needed

See `docs/tweaks.md` for the running list of known issues and planned improvements.