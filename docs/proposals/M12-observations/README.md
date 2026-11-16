# M12 checkpoint 2: observation-model review

Each row is one channel link for an in-scope component: a fault mode's `observable_by` (scope O observed_by or
threshold event), an M8 `named_by` channel, or the declared CNC accelerometer. Every row has a proposed verdict and a
physical reason. **?** marks the ones I'm unsure of. An accepted row becomes an `[[observation]]` entry in that
machine's machine.toml, drafted in this folder. The drafts were dry-run through parameterize step f on each
machine's latest artifacts, offline: every quote is verbatim on its page, and no gate fails.

Pages are physical, with the printed label in brackets.

## eversys_coffee_machine

| Component | Channel | Link | Verdict | Reason |
|---|---|---|---|---|
| water_filter | error_message | threshold event (water_filter_exhaustion) | **accept**: threshold_event | W-014 p114 (114): "Please change the water filter. Predefined water amount has been reached." This is the filter reaching its limit, and the filter is replaced on it. |
| top_piston | error_message | threshold event (top_piston_residue_buildup) | **accept**: threshold_event | W-087/W-088 p125 (125): "Top piston zero position failure … check the top piston (screen and rod) for coffee residue and clean it." Residue at its limit blocks the zero position. |
| coffee_boiler | coffee_boiler_temperature | named_by | **reject** | A controlled temperature. Scale insulates the heater, and the controller makes up for it by heating longer, so the reading stays at its setpoint. Heating time would show scale; no channel reports it. |
| steam_boiler | steam_boiler_pressure | named_by | **reject** | Controlled. S-100 p110 (110): below 0.8 bar "the boiler starts to heat to the nominal value". |
| milk_system | milk_temperature | named_by | **reject** | A product setting, "Milk Temperature Warm" p153 (153), heated to that setting. Residue in the milk lines does not move it. |
| steam_wand | steam_wand_temperature | named_by | **reject ?** | The wand heats the milk to a set temperature. A clogging wand delivers less steam, so it takes longer to get there, but it still reaches the setting. No document says how this channel behaves. |

`error_message` is a bool: it shows *a* message, not *which* one, so W-014 and W-087/W-088 cannot be told apart.
A category channel of W-codes would be an understand follow-up; it is logged, not changed in M12.

## air_compressor

| Component | Channel | Link | Verdict | Reason |
|---|---|---|---|---|
| oil_separator_element_os | pressure_difference_over_oil_separator | observable_by | **accept**: linear | A clogging element is a finer restriction, so the drop rises. y_limit = 1 bar, manual, p29 (27): "Or when the pressure drop over the separator exceeds 1 bar." y_healthy 0.1–0.5 bar is a judgement: only the normal reading "Below 1 bar" is printed (p33, label 31). |
| oil_separator_element_os | service_warning | threshold event | **accept**: threshold_event | p9 (7): the regulator monitors the oil separator, "Each input is compared to programmed limits (temperatures, pressures and/or running hours)", and warns when a limit is exceeded. |
| oil_separator_element_os | air_outlet_pressure | observable_by | **reject** | Regulated: it "Modulates between programmed unloading and loading pressures" (p33, label 31). |
| oil_cooler_co | compressor_element_outlet_temperature | observable_by | **accept ?**: linear | Above ~55 °C "all the oil flows through the oil cooler" back to the element (p7, label 5), so a fouled cooler raises the element outlet temperature. y_healthy 75–85 °C, manual: "Approx. 60 degrees celsius above cooling air temperature" (p33, label 31) at the 20 °C reference inlet (p35, label 33). **y_limit 90–130 °C is a judgement:** the shut-down warning level is "programmed" (p9, label 7) and never printed. |
| air_cooler_ca | compressor_element_outlet_temperature | observable_by | **reject** | Wrong place in the flow: the air goes "via minimum pressure valve (Vp), air cooler (Ca) and condensate trap" (p6, label 4), downstream of the element. Its fouling would show on air outlet temperature or dewpoint, and no fault mode links those. |

## cnc_spindle

| Component | Channel | Link | Verdict | Reason |
|---|---|---|---|---|
| angular_contact_bearings | spindle_housing_vibration (declared) | declared | **accept**: linear | Rolling-contact damage raises housing vibration. Thesis Figure 14, p30 (20), SIS 728000-1 / ISO TR 17243-1, motor spindles ≤ 15 kW, measured on the housing (p28, label 18), LTSC: A/B 0.7, B/C 1.1, C/D 1.8 mm/s RMS. y_limit 1.1–1.8 (zone C), y_healthy 0.3–0.7 (zone A; 0.3 from p113, label 103). Literature, read from the figure and checked by eye against the rendered page. |
| cylindrical_roller_bearings | spindle_housing_vibration (declared) | declared | **accept**: linear | Same channel and zone priors. The rear bearing is further from a front-mounted sensor, so the same zone C prior is if anything generous. |
| spindle_motor | first_spindle_motor_temperature, spindle_motor_temperature_information | named_by | **reject** | Insulation ageing is *driven* by temperature (the Arrhenius condition); it does not change the temperature. Feeding these to the template's temperature condition is a later item. |
| spindle_motor | motor_speed, spindle_motor_speed_data | named_by | **reject** | Commanded speed, held by the drive. |
| spindle_motor | spindle_motor_torque_data | named_by | **reject** | Load torque is set by the cut; thermal ageing of insulation does not change it. |

**Why LTSC and not STSC:** the thesis says STSC (acceleration RMS, 2–10 kHz) "reflects better problems
associated with bearing condition" (p29, label 19). But STSC's zone boundaries depend on the speed class and appear only as colour
bands in Figure 14, with no readable numbers. LTSC's boundaries are printed and are the same for every speed class.

**Sum on a shared channel:** both bearings act on one sensor, so
y = y_healthy + Σᵢ (y_limit,ᵢ − y_healthy) · clip(Dᵢ/D_limit,ᵢ, 0, 1). With both bearings at their limits the
reading is in zone D.

## Defaults applied to every accepted linear channel

- **Noise:** a lognormal whose median is 5% of the smallest |y_limit − y_healthy| (prior medians), log-sd 0.5,
  source `default`, flagged. No document gives measurement repeatability.
- **Judged values** score `judgement_confidence_base` (0.3) times their width term, and are flagged.
