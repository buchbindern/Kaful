"""
phases/phase7_estimate/__init__.py
------------------------------------
Phase 7: State estimation and RUL prediction.

Steps:
    a — load simulator data and build observations per usage profile
    b — run particle filter for state estimation per profile
    c — run Monte Carlo RUL prediction per profile

Each step saves its output to disk and skips if already done.
Results are saved per usage profile.

Call from run.py:
    from phases.phase7_estimate import run
    rul = run(cfg)
"""

from phases.phase7_estimate.step_a_load import run as step_a
from phases.phase7_estimate.step_b_particle_filter import run as step_b
from phases.phase7_estimate.step_c_rul import run as step_c


def run(cfg: dict) -> dict:
    """
    Run all phase 7 steps in order.

    Args:
        cfg: result of get_machine_config()

    Returns:
        dict of {profile_name: rul_results}
    """
    print(f"\n{'='*52}")
    print(f"Phase 7 — Estimate: {cfg['machine_id']}")
    print(f"{'='*52}")

    profiles    = step_a(cfg)
    estimations = step_b(cfg, profiles)
    rul         = step_c(cfg, estimations)

    print(f"\n✓ Phase 7 complete — {len(rul)} profiles estimated.")
    return rul