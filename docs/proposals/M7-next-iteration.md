# M7: next paid iteration per machine (proposal, 2026-11-08)

For Natalie's go. Nothing here is applied. The exit time-box leaves eversys 2 paid iterations and
CNC 1.

## 1. CNC 20261108T213530Z-ecb5ac: what happened

sensor_channels v5 (max_tokens 32000) fixed the truncation: sample 0 wrote 17,241 output tokens.
The run then failed on validation:

| Sample | Attempt 1 | Attempt 2 (retry) |
|---|---|---|
| 0 | invalid: `first_spindle_position_data`: numeric channels need a unit | valid, 45 channels |
| 1 | invalid: `first_spindle_error_state`: same | invalid: `motor_speed`: same; the run stops |

Cost: $0.0317 (components replayed from cache at $0).

**The retry message invites made-up units.** The validator says "('1' if dimensionless)". Sample
0's valid retry set unit `1` on every p730 (label 706) trace value, including motor speed, motor
current, winding temperature and DC link voltage. p730 is the SERVO GUIDE trace list, and it prints
no units. So a valid reply here is worse than the invalid one.

### Channel precision (sample 0's valid retry, matched by hand against the golden rows)

- 45 channels. They hit 15 of the 22 golden channels: recall 0.68. Leaving out the two vibration
  rows, which only the KTH thesis supports, it is 15 of 20 (0.75).
- Precision is 0.33 when each golden channel counts once (15 of 45). It is 0.58 if second-spindle
  copies and repeated sources of the same value also count as correct (26 of 45).
- Where the 30 extra rows come from:

| Kind | Rows | Examples |
|---|---|---|
| Second-spindle copies from the p200 (label 176) diagnosis table | 6 | `second_spindle_load_meter`, `second_spindle_motor_temperature` |
| The same value from a second page | 5 | `spindle_speed` / `motor_speed` (p730) beside `first_spindle_speedometer` (p200) |
| Position and interpolation pulses (p128 (label 104)), not in the golden file | 7 | `spindle_interpolation_pulse`, `spindle_synchronous_error_width_max` |
| Parameters, not data (p240 (label 216)) | 2 | `disturbance_torque_command_frequency`, `torque_command_gain` |
| Golden `not_channel` rows | 2 | `motor_current_amplitude` (golden motor_current), `speed_deviation` (velocity_error) |
| Other p243 (label 219) / p217 (label 193) values | 8 | sensor feedback data, `speed_detection_signal`, position data |

So 34 to 45 channels is over-extraction, mostly copies and pages listing the same value twice. It is
not a lot of channels nobody needs.

- Missed: `motor_voltage`, `acceleration_deceleration_time`, `orientation_time`, `operation_mode`,
  `spindle_warning_state`, and the two vibration rows. `spindle_state_error` is in sample 1 only
  (`first_spindle_error_state`).

### Proposed fixes (one CNC iteration left)

1. **Unit rule in code, not a retry.** The output model stops raising "numeric channels need a
   unit". `_type_fields` already drops a numeric channel without a unit and logs it. Change that
   to keep the channel with `unit = null`, `unit_status = "not_printed"`, and no range. M8 or the
   user then supplies the unit. A dropped motor_speed is a lost observation, and a made-up `1` is a
   wrong unit. This changes the SensorChannel format (`unit_status`), which needs your go. The
   simpler variant drops and logs, with no format change, but it loses the p730 values.
   Either way, the error text loses "('1' if dimensionless)".
2. **sensor_channels v6: one sentence on copies.** "When a table lists the same value for several
   spindles, axes or units, report it once for this machine. When two pages list the same value,
   report it once, citing both pages." Add a CNC eval case built from p200 and p730, with
   `no_extra_channels` and a check that no name starts with `second_`.
3. **Cost:** evals 2 runs, about $0.012. CNC rerun about $0.03, with components cached and fault
   modes fresh. Cap $0.06.

The bearings (machine.toml `[[components]]`) were not exercised: fault_modes never ran. The cached
components samples for the Fanuc-only context all name `spindle_motor` (3 of 3), so the declared
parent resolves. `position_coder` is in none of them, which fails the carriers floor (3 golden
`yes` modes).

## 2. eversys: components v5 (carriers: milk_system)

The error and warning lists give a module or unit for each entry:
- p102 (label 102): Bean hopper, Hydraulic module, Milk module, Brewing module;
- p103 (label 103): Grinder module, Steam module.

Run 21c0ec never extracted `milk_system`, so the milk residue mode had no component. The canonical
merge already joins `milk_module` with `milk_system` (generic head "module").

**components v5 (system.md, one line added under what a component is):**

```
- An error or warning list that names a module or unit for each entry ("Hydraulic module", "Milk
  module", "Grinder module") names the machine's main assemblies. Report each one as a top-level
  component, unless the documentation places it inside another part, and put the parts the
  documentation places in it under it.
```

New eval case `eversys_modules`: the raw text of p102 and p103. Check `module_components`:
`components[].id`, after the generic-head rule, contains_all [hydraulic, milk, grinder, brewing,
steam]. Threshold 1.0.

**Cost:** evals 2 runs (tp200 plus the eversys cases), about $0.02. eversys whole-manual rerun with
components fresh, about $0.07. Cap $0.10. This is eversys iteration 1 of 2.
