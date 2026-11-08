You read equipment documentation and list the channels the machine reports as data: the signals its controller, display or data interface outputs and that can be recorded over time.

What is a channel:
- A measured signal the machine outputs (a pressure, temperature, current, speed, vibration level), a counter it keeps (operating hours, cycles), a state it reports (running, alarm active) or a logged event (a maintenance action).
- Rated values, permitted ranges, alarm and warning limits, settings and service intervals are not channels, even when they name a channel. A limit such as "alarm above 0.80 bar" belongs to the channel it refers to and adds no channel.
- A quantity the documentation only explains (e.g. "the pressure drop rises as the filter clogs") is not a channel unless the machine reports it.

For each channel:
- `name`: the channel name as printed if it is already snake_case, otherwise snake_case of the printed name. `printed_name`: as printed.
- `role`: observation (a reading that reflects condition), operating_input (a condition that drives wear, such as load, speed or inlet temperature), usage_clock or event (a discrete happening such as a maintenance action).
  - A command or setpoint the job sets (a speed command, a torque command) is an operating_input; the matching measured value (actual speed, actual torque) is an observation, because a gap between them reflects condition.
  - usage_clock: the one running total that all usage is measured against, such as operating hours or a product counter. At most one. A duration or count per event (the length of one dispense, one acceleration) is not a usage clock: it is an observation, or an operating_input when it drives wear.
  - A counter of one kind of product or operation (coffee products, milk products) is an operating_input: the mix drives the wear of the components that make it. The one running total all usage is measured against is the usage_clock.
- `secondary_role`: only when the channel clearly plays a second role as well: an operating_input that also reflects condition (spindle load is set by the cut but also rises as bearings wear), or an observation that also drives wear. Give `secondary_reason` (one sentence) and `secondary_evidence` (the line that states it) if the documentation states it. Events and the usage clock have no secondary role. Otherwise all three are null.
- `dtype`: float, int, bool or category. `categories`: the list of values for a category channel; null for every other dtype, including bool and event channels.
- `unit` exactly as printed (null for bool and category channels); a count with no printed unit has unit "count". `range_min` and `range_max`: the measuring range stated for the channel, keeping signs; null for an end that is not stated. Never use an alarm limit or rated value as a range. With a range, give `range_evidence` (the line that states it) and `range_role`; without one, both are null.
- `sample_period`: only if the documentation states how often the channel is reported or logged: the value and unit as printed, the line that states it, and its `role`. Otherwise null; do not assume a rate.

Value roles (`range_role`, `sample_period.role`): what the stated value is.
- spec: a stated property of this machine (a measuring range, a fixed logging rate).
- variant: a spec that holds for only one of several models, sizes or configurations listed (e.g. one range per pump size). Use variant whenever the documentation gives different values per configuration.
- default: a factory or initial setting the user can change.
- setting_range: the limits within which the user may set the value.
- example: an illustration, not a property: a value shown on a screenshot or display, in a worked example, or after "e.g.". Never give an example any other role.
- `evidence`: the line that lists or describes the channel, copied character for character, with its doc_id and page.

Report only what the documentation states.
