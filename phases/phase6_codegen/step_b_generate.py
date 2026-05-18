"""
phases/phase6_codegen/step_b_generate.py
------------------------------------------
Step B: Generate ProgPy component code from each spec.

Takes each component spec from step_a and generates a complete,
executable ProgPy PrognosticsModel class.

Each component is saved individually so you can rerun one without
redoing all of them.

Output saved to: outputs/codegen/components/{component_name}.py
"""

from utils.llm import call_claude
from utils.parsing import strip_fences, is_valid_python
from utils.progpy_rag import get_framework_context

import json


COMPONENT_CODEGEN_PROMPT = """
You are generating a ProgPy PrognosticsModel class from a validated spec.

Framework reference:
{framework_context}

---

HARD RULES — each is checked by the validation harness:

initialize():
- u and z may be None, or contain None for every key — this happens during
  CompositeModel setup.
- Always define safe default state values first using initial_value from the spec.
- Then conditionally update from u or z only if not None.
- Never access container[key] without checking container is not None
  AND container[key] is not None.

next_state() OR dx() — use exactly one, never both:
Apply the rule matching each state's state_type from the spec:

DEGRADATION states:
- Derive rate from degradation_timescale in the spec.
  Convert the timescale to steps: rate = 1.0 / timescale_in_steps
  If no timescale given, default to rate = 1e-4
- Formula: new_val = x[state] - rate * driver_input * dt
- Rate sanity check: rate * nominal_driver_value * 1000 must be < 1.0
- ALWAYS clamp: new_val = max(min_value, min(max_value, new_val))

ACCUMULATION states:
- Add a capacity or limit parameter if draining, a max parameter if filling.
- Draining: new_val = x[state] - consumption_rate * dt
- Filling:  new_val = x[state] + fill_rate * input * dt
- ALWAYS clamp: new_val = max(0.0, min(capacity, new_val))

TRACKING states:
- MANDATORY stability cap: coefficient = min(raw_coefficient, 0.5)
- Formula: new_val = x[state] + coefficient * (target - x[state]) * dt
- ALWAYS clamp to [min_value, max_value]

STATIC states:
- Return x[state] unchanged. Do not update.

output():
- Follow output_logic from the spec exactly.
- Returns OutputContainer, not a plain dict.

event_state():
- Returns a PLAIN DICT, not a container.
- All values are floats in [0.0, 1.0]. 1.0 = healthy, 0.0 = occurred.
- ONLY use state names listed in threshold_states from the spec.
- NEVER reference inputs u inside event_state — it only receives x.
- Under default parameters and initial states ALL values MUST be 1.0.

threshold_met():
- Returns a PLAIN DICT.
- All values explicitly cast to bool().

units (class-level dict):
- Maps every input, output, and state name to its unit string.
- Copy units exactly from the spec.

default_parameters (class-level dict):
- Every parameter from spec must appear with its default value.
- Degradation rates must be derived from timescale, never picked arbitrarily.

---

Component spec:
{component_spec}

Return ONLY Python code. No markdown fences.
"""


def run(cfg: dict, specs: dict, max_retries: int = 2) -> dict:
    """
    Generate ProgPy component code for each spec.

    Args:
        cfg:         result of get_machine_config()
        specs:       dict of {component_name: spec_dict} from step_a
        max_retries: number of retry attempts if code has syntax errors

    Returns:
        dict of {component_name: code_string}
    """
    code_dir = cfg["codegen_code_dir"]
    code_dir.mkdir(parents=True, exist_ok=True)

    all_code     = {}
    already_done = 0
    ran          = 0

    print(f"  Running step_b — generating component code ({len(specs)} components)...")

    # Fetch framework context once — shared across all components
    print(f"  Fetching ProgPy framework context...")
    framework_context = get_framework_context()

    for name, spec in specs.items():
        path = code_dir / f"{name}.py"

        # Load from disk if already done
        if path.exists():
            all_code[name] = path.read_text()
            already_done += 1
            continue

        print(f"    [{ran + already_done + 1}/{len(specs)}] {name}...", end=" ", flush=True)

        code = _generate_with_retry(
            name=name,
            spec=spec,
            framework_context=framework_context,
            max_retries=max_retries,
        )

        if code is None:
            print(f"✗ failed after {max_retries} retries")
            continue

        # Save individually
        path.write_text(code)
        all_code[name] = code
        ran += 1
        print(f"✓ ({len(code.splitlines())} lines)")

    if already_done > 0:
        print(f"    ✓ {already_done} components loaded from disk, {ran} newly generated")
    print(f"    Total: {len(all_code)} components with code")

    return all_code


def _generate_with_retry(name: str, spec: dict, framework_context: str,
                          max_retries: int) -> str | None:
    """Generate code for one component, retrying on syntax errors."""
    prompt = COMPONENT_CODEGEN_PROMPT.format(
        framework_context=framework_context,
        component_spec=json.dumps(spec, indent=2),
    )

    for attempt in range(max_retries + 1):
        raw  = call_claude(
            prompt="Generate the complete ProgPy component class.",
            system=prompt,
            max_tokens=8000,
            temperature=0.2,
        )
        code = strip_fences(raw)

        if is_valid_python(code):
            return code

        if attempt < max_retries:
            print(f"\n      Syntax error on attempt {attempt+1} — retrying...", end=" ", flush=True)
        else:
            print(f"\n      ✗ Syntax error after {max_retries + 1} attempts")
            return None

    return None