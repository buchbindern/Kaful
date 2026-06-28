- **State estimation is open loop.** The ensemble is propagated forward under the
  loading profile; it does not condition on incoming observations. Bayesian updating
  (likelihood + resampling) is deliberate future work — closing the loop requires real
  machine telemetry, which deployment provides.
- **No historical failure data.** Degradation is physics driven from the simulator, so
  RUL reflects modeled wear, not fleet-calibrated failure statistics.