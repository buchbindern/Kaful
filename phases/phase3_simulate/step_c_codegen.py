"""
phases/phase3_simulate/step_c_codegen.py
------------------------------------------
Step C: Generate the simulator code.

Takes the schema, manual context, and simulator plan from previous steps
and generates a complete, executable Python simulator.

Output saved to: outputs/simulate/simulator.py
"""

import json

from utils.llm import call_claude
from utils.parsing import strip_fences, is_valid_python


DATA_SIMULATOR_PROMPT = """# Equipment Time Series Simulator Generator

You are an expert in industrial IoT, time series data generation, and equipment degradation modeling.

Your task is to write a complete Python simulator that generates realistic event-level telemetry data for a specific piece of equipment.

---

## Equipment Type

{machine_type}

---

## Schema

{sensor_measurements}

Each row contains:
- field_name
- data_type
- unit
- typical_range
- description

All generated data MUST match this schema exactly:
- no missing fields
- no extra fields
- exact field names

---

## Manual Context

The following context was extracted from the equipment manual.

Use it to inform:
- operating ranges
- process behavior
- subsystem interactions
- degradation mechanisms
- maintenance behavior
- realistic operating environments

{manual_context}

---

## Simulator Plan

{simulator_plan}

The simulator plan is the source of truth.

You MUST:
- implement all subsystems, state variables, and operation types
- follow all field_generation_rules
- enforce all null_field_rules
- implement degradation, maintenance, and failure logic exactly as described

Do NOT redesign the system.
Do NOT invent extra subsystems, fields, or operation types unless strictly required
to implement the provided plan and schema.

---

## Architecture

Structure the simulator into four layers:

### 1. Configuration dataclass
- define all constants here
- no magic numbers in logic
- include wear rates, thresholds, noise parameters, maintenance restoration factors,
  and operating assumptions

### 2. State dataclasses
- one per subsystem
- state persists across events
- include degradation, buildup, thermal, calibration drift, and usage state as appropriate

### 3. Usage profiles
- define realistic operating environments
- include operating hours, event intensity, operation mix, maintenance schedule
- event arrivals must follow a Poisson process

### 4. Event generation functions
- pure functions
- inputs: (state, config, rng, operation context)
- outputs: dict of schema field values
- compute outputs first, then update persistent state
- return one complete event record

---

## Time-Scale Separation Requirements

The simulator must model TWO distinct time scales:

### Short-term dynamics
- warm-up, cool-down, per-event variability, within-day usage stress
- these stabilize quickly, typically within hours or days

### Long-term dynamics
- cumulative wear, buildup, drift, efficiency loss, calibration shift
- these MUST evolve across the entire simulation
- warm-up effects MUST NOT substitute for degradation

After short-term stabilization, at least one core process metric must continue
drifting due to long-term degradation.

---

## Long-Horizon Degradation Requirements

At least one core process metric must:
- appear in most relevant events
- depend on persistent degradation state
- show gradual drift across the FULL simulation window
- NOT plateau early within the first 10-20% of the simulation unless maintenance
  explicitly causes temporary recovery

---

## Degradation-to-Sensor Coupling

For each degrading subsystem:
- degradation must affect at least one measured or outcome-related field
- the effect must include BOTH shift in expected value AND increase in variance
- at least one core measured field must show meaningful difference between
  first 20% and last 20% of the simulation

---

## Nonlinear Degradation Requirement

Degradation effects must not be purely linear.
Use realistic behavior such as:
- mild early effects that become stronger later
- interaction effects between wear and workload
- accelerating or slowing degradation depending on subsystem condition

---

## Variance Evolution Requirement

As degradation increases:
- variability of key measured fields must increase
- noise level should depend on subsystem condition, not remain constant
- at least one key process metric must be noticeably noisier late vs early

---

## Local Irregularity Requirement

Include realistic short-window irregularity:
- small local reversals, short plateaus, temporary deviations
- event-to-event variation that is not perfectly uniform
- workload-dependent fluctuations

---

## Failures

- triggered by degraded system behavior or sensor values leaving bounds
- NOT based on arbitrary random flags
- initial success rate ~95%+ for a healthy machine
- success rate deteriorates gradually with wear

---

## Maintenance

Must:
- be triggered by degradation, buildup, thresholds, schedules, or usage logic
- partially restore system state (NOT full reset)
- create visible step changes in at least one key metric
- In a typical 30-day simulation with moderate/high usage, at least one maintenance
  event should occur

---

## Field Requirements

- control_input: sampled, selected, or configured per event
- measured_sensor: computed from process physics, state, and noise
- duration: derived from process timing and system condition
- outcome: determined from system behavior, thresholds, and event results
- metadata: contextual event information

Requirements:
- all schema fields must be present in every output row
- use exact field names
- use None for non-applicable fields
- NEVER use zero as a placeholder for missing or inapplicable values

---

## Randomness

- use np.random.default_rng()
- pass rng explicitly through simulator functions
- do not use global randomness

---

## Output

Return code that produces:

1. Events DataFrame — one row per event, all schema fields
2. Maintenance log DataFrame — one row per maintenance event
3. Main execution block (if __name__ == "__main__") that:
   - runs the simulator for all defined usage profiles
   - prints a concise summary
   - saves CSV files

---

## Code Quality

The code must be:
- fully executable without modification
- complete, well-structured, readable
- deterministic given a fixed random seed

Do NOT include TODOs, placeholder comments, markdown fences, or explanatory prose.

Return ONLY complete Python code."""


def run(cfg: dict, schema: list[dict], manual_context: str, plan: dict) -> str:
    """
    Generate the simulator code.

    Args:
        cfg:            result of get_machine_config()
        schema:         final schema from phase 2
        manual_context: formatted context string from step_a
        plan:           simulator plan from step_b

    Returns:
        simulator code as a string
    """
    output_path = cfg["sim_step_c_code"]

    # Load from disk if already done
    if output_path.exists():
        print("  ✓ step_c already done — loading from disk")
        return output_path.read_text()

    print("  Running step_c — generating simulator code...")
    print("  (This may take a minute — generating ~500 lines of Python)")

    prompt = DATA_SIMULATOR_PROMPT.format(
        machine_type=plan.get("machine_type", ""),
        sensor_measurements=json.dumps(schema, indent=2),
        manual_context=manual_context,
        simulator_plan=json.dumps(plan, indent=2),
    )

    raw = call_claude(
        prompt="Generate the complete simulator code.",
        system=prompt,
        max_tokens=16000,
        temperature=0.2,
    )

    code = strip_fences(raw)

    if not is_valid_python(code):
        raise ValueError(
            f"step_c: generated code has syntax errors.\n"
            f"First 500 chars:\n{code[:500]}"
        )

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)

    print(f"    ✓ Generated {len(code.splitlines())} lines of Python")
    print(f"    Saved → {output_path.name}")

    return code