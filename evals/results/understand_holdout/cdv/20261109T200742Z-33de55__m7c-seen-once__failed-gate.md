# cdv run 20261109T200742Z-33de55 (M7c; SEEN ONCE, NOT A CLEAN HOLD-OUT; FAILED GATE: units 'SLM', 'SCCM', 'mTORR', 'MTOR', 'hh:mm:ss'; scored outside the CLI): disagreements with the golden labels, not pre-filled, not adjudicated

Golden file sha256 864a54614ff1. `proposed` is Claude's verdict and reason, `unsure` marks the ones to review; verdict: `run` (fix the label) or `label`.

| kind | item | run says | labels say | run cites | proposed | unsure | verdict |
|---|---|---|---|---|---|---|---|
| channel | abort_taken | a channel (event) | no row | m410_01_dpc p.20 (label 20) |  |  |  |
| channel | active_alarm | a channel (event) | no row | m210_01_operator p.12 (label 12) |  |  |  |
| channel | ai_2_torchact | a channel (observation) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | ai_t_torchact | a channel (observation) | no row | m220_383_02_process_manual p.90 (label 90) |  |  |  |
| channel | air_pressure_switch | not extracted | channel (event) | m210_01_operator p13 |  |  |  |
| channel | alarm_active | a channel (event) | no row | m330_01_gas_cabinet p.29 (label 29) |  |  |  |
| channel | analog_output_setpoint | a channel (operating_input) | no row | m410_01_dpc p.18 (label 18) |  |  |  |
| channel | ao_1_n2 | a channel (operating_input) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | ao_2_o2 | a channel (operating_input) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | ao_3_o2_low | a channel (operating_input) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | ao_4_h2 | a channel (operating_input) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | ao_7_n2_low | a channel (operating_input) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | ao_p_n2 | a channel (observation) | no row | m220_383_02_process_manual p.90 (label 90) |  |  |  |
| channel | base_pressure | no such secondary role | secondary operating_input | m330_01_gas_cabinet p.29 (label 29) |  |  |  |
| channel | boat_alarm | a channel (event) | no row | m410_01_dpc p.8 (label 8) |  |  |  |
| channel | boat_position | role observation | role operating_input | m410_01_dpc p.22 (label 22) |  |  |  |
| channel | boat_speed | a channel (operating_input) | no row | m410_01_dpc p.18 (label 18) |  |  |  |
| channel | branch_taken | a channel (event) | no row | m410_01_dpc p.19 (label 19) |  |  |  |
| channel | bubbler_temperature | a channel (observation) | no row | m320_01_furnace p.82 |  |  |  |
| channel | bubbler_temperature_input | a channel (observation) | no row | m220_383_02_process_manual p.82 (label 82) |  |  |  |
| channel | current_alarms | a channel (event) | no row | m220_383_02_process_manual p.76 (label 76) |  |  |  |
| channel | current_step | a channel (observation) | no row | m210_01_operator p.16 (label 16) |  |  |  |
| channel | dcs_flow | no such secondary role | secondary observation | m220_383_02_process_manual p.42 (label 42) |  |  |  |
| channel | di_1_press_n2 | a channel (event) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | di_2_pressair | a channel (event) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | di_3_torch_ok | a channel (event) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | di_4_ratio_ok | a channel (event) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | di_5_flame_ok | a channel (event) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | di_6_temphous | a channel (event) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | di_7_temp_scr | a channel (event) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | di_8_excess | a channel (event) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | di_9_fancontr | a channel (event) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | digital_output | a channel (operating_input) | no row | m410_01_dpc p.18 (label 18) |  |  |  |
| channel | digital_output_state | a channel (operating_input) | no row | m410_01_dpc p.18 (label 18) |  |  |  |
| channel | digital_pressure_switch_pressure | a channel (observation) | no row | m330_01_gas_cabinet p.13 (label 13) |  |  |  |
| channel | do_1_n2 | a channel (operating_input) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | do_2_o2 | a channel (operating_input) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | door_closed | not extracted | channel (event) | m220_383_02_process_manual p40 |  |  |  |
| channel | excess_temperature | not extracted | channel (event) | m210_01_operator p13 |  |  |  |
| channel | exhaust_flow_alarm | not extracted | channel (event) | m210_01_operator p13 |  |  |  |
| channel | gas_o2_low_flow | a channel (observation) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | gas_pressure | a channel (observation) | no row | m330_01_gas_cabinet p.27 (label 27) |  |  |  |
| channel | leak_detection | not extracted | channel (event) | m210_01_operator p13 |  |  |  |
| channel | light_tower_green | a channel (event) | no row | m210_01_operator p.12 (label 12) |  |  |  |
| channel | light_tower_red | a channel (event) | no row | m210_01_operator p.12 (label 12) |  |  |  |
| channel | light_tower_yellow | a channel (event) | no row | m210_01_operator p.12 (label 12) |  |  |  |
| channel | limit_alarm | not extracted | channel (event) | m210_01_operator p13 |  |  |  |
| channel | mfc_actual_flow | another name for gas_flow | one item | m330_01_gas_cabinet p.27 (label 27) |  |  |  |
| channel | mfc_flow_setpoint | a channel (operating_input) | no row | m410_01_dpc p.12 (label 12) |  |  |  |
| channel | mfc_gas_flow | a channel (observation) | no row | m330_01_gas_cabinet p.27 (label 27) |  |  |  |
| channel | mfc_setpoint | a channel (operating_input) | no row | m410_01_dpc p.12 (label 12) |  |  |  |
| channel | n2_flow | another name for gas_flow | one item | m220_383_02_process_manual p.40 (label 40) |  |  |  |
| channel | n2_pressure_switch | not extracted | channel (event) | m210_01_operator p13 |  |  |  |
| channel | n2_supply_pressure | a channel (observation) | no row | m330_01_gas_cabinet p.13 (label 13) |  |  |  |
| channel | nh3_flow | another name for gas_flow | one item | m220_383_02_process_manual p.41 (label 41) |  |  |  |
| channel | o2_flow | another name for gas_flow | one item | m220_383_02_process_manual p.59 (label 59) |  |  |  |
| channel | paddle_actual_value | a channel (observation) | no row | m320_01_furnace p.32 (label 32) |  |  |  |
| channel | powerpack_temperature | not extracted | channel (event) | m210_01_operator p13 |  |  |  |
| channel | pressure | another name for process_pressure | one item | m220_383_02_process_manual p.40 (label 40) |  |  |  |
| channel | process_pressure | another name for process_pressure | one item | m410_01_dpc p.14 (label 14) |  |  |  |
| channel | process_recipe_running | a channel (event) | no row | m210_01_operator p.16 (label 16) |  |  |  |
| channel | process_time | a channel (observation) | no row | m210_01_operator p.18 (label 18) |  |  |  |
| channel | process_value | a channel (observation) | no row | m210_01_operator p.18 (label 18) |  |  |  |
| channel | remaining_step_time | a channel (observation) | no row | m410_01_dpc p.19 (label 19) |  |  |  |
| channel | safety_setpoint | a channel (operating_input) | no row | m220_383_02_process_manual p.88 (label 88) |  |  |  |
| channel | scavenger_pressure | a channel (observation) | no row | m320_01_furnace p.11 (label 11) |  |  |  |
| channel | sih4_flow | another name for gas_flow | one item | m220_383_02_process_manual p.60 (label 60) |  |  |  |
| channel | spike_temp_2 | a channel (observation) | no row | m220_383_02_process_manual p.90 (label 90) |  |  |  |
| channel | step_number | a channel (observation) | not_channel | m410_01_dpc p.19 (label 19) |  |  |  |
| channel | step_time | a channel (observation) | no row | m210_01_operator p.18 (label 18) |  |  |  |
| channel | temp_power_alarm_zone | a channel (event) | no row | m220_383_02_process_manual p.77 (label 77) |  |  |  |
| channel | teos_bubbler_temperature | a channel (observation) | no row | m330_01_gas_cabinet p.29 (label 29) |  |  |  |
| channel | torch_flame_present | a channel (event) | no row | m410_01_dpc p.15 (label 15) |  |  |  |
| channel | vacuum_atmosphere_switch | a channel (event) | no row | m330_01_gas_cabinet p.29 (label 29) |  |  |  |
| channel | vacuum_fail | not extracted | channel (event) | m220_383_02_process_manual p40 |  |  |  |
| channel | wait_alarm | a channel (event) | no row | m410_01_dpc p.16 (label 16) |  |  |  |
| channel | water_cooling | not extracted | channel (event) | m220_383_02_process_manual p40 |  |  |  |
| channel | zone_temperature | not extracted | channel (operating_input) | m320_01_furnace p18 |  |  |  |
| component | analog_io_board | a component | no row | m410_01_dpc p.12 (label 12) |  |  |  |
| component | boat_loader | not extracted | child | m210_01_operator p10 |  |  |  |
| component | bubbler_system | a component | not_a_component | m330_01_gas_cabinet p.14 (label 14) |  |  |  |
| component | cassettes_baffles | parent process_tubes | parent load_station | m220_383_02_process_manual p.38 (label 38) |  |  |  |
| component | communication_board | a component | no row | m410_01_dpc p.13 (label 13) |  |  |  |
| component | control_system | a component | not_a_component | m320_01_furnace p.8 (label 8) |  |  |  |
| component | cooling_fans | not extracted | child | m320_01_furnace p38 |  |  |  |
| component | digital_io_board | a component | no row | m410_01_dpc p.12 (label 12) |  |  |  |
| component | digital_process_controller | a component | not_a_component | m410_01_dpc p.7 (label 7) |  |  |  |
| component | door_o_ring | parent furnace | parent flange | m220_383_02_process_manual p.37 (label 37) |  |  |  |
| component | external_torch | a component | not_a_component | m330_01_gas_cabinet p.16 (label 16) |  |  |  |
| component | fast_cooldown_heat_exchanger | a component | no row | m320_01_furnace p.13 (label 13) |  |  |  |
| component | fine_metering_valve | not extracted | child | m330_01_gas_cabinet p11 |  |  |  |
| component | flange | not extracted | child | m320_01_furnace p10 |  |  |  |
| component | flowmeter | a component | not_a_component | m330_01_gas_cabinet p.12 (label 12) |  |  |  |
| component | gas_line_regulator | a component | no row | m330_01_gas_cabinet p.31 (label 31) |  |  |  |
| component | gas_line_valve | a component | no row | m330_01_gas_cabinet p.31 (label 31) |  |  |  |
| component | heat_exchanger | another name for cooling_unit | one item | m320_01_furnace p.38 (label 38) |  |  |  |
| component | inline_filter | a component | no row | m330_01_gas_cabinet p.25 (label 25) |  |  |  |
| component | inline_gas_filter | a component | no row | m330_01_gas_cabinet p.12 (label 12) |  |  |  |
| component | inline_gas_filters | not extracted | child | m330_01_gas_cabinet p12 |  |  |  |
| component | io_interconnection_board | a component | no row | m410_01_dpc p.14 (label 14) |  |  |  |
| component | loadstation | another name for load_station | one item | m210_01_operator p.10 (label 10) |  |  |  |
| component | main_power_cabinet | a component | no row | m210_01_operator p.5 (label 5) |  |  |  |
| component | main_switch_assembly | a component | not_a_component | m320_01_furnace p.10 (label 10) |  |  |  |
| component | manual_shutoff_valves | not extracted | child | m330_01_gas_cabinet p11 |  |  |  |
| component | mass_flow_controller | parent gas_cabinet | parent gaspanel | m220_383_02_process_manual p.7 (label 7) |  |  |  |
| component | mass_vac_dust_trap | parent None | parent vacuum_pump_system | m320_01_furnace p.40 (label 40) |  |  |  |
| component | o_rings | another name for o_rings | one item | m320_01_furnace p.41 (label 41) |  |  |  |
| component | pneumatic_valves | not extracted | child | m330_01_gas_cabinet p11 |  |  |  |
| component | power_supply | not extracted | child | m320_01_furnace p10 |  |  |  |
| component | powersupply_board | a component | no row | m410_01_dpc p.14 (label 14) |  |  |  |
| component | pressure_interface_board | a component | not_a_component | m410_01_dpc p.12 (label 12) |  |  |  |
| component | process_tube | not extracted | child | m320_01_furnace p10 |  |  |  |
| component | process_tubes | a component | no row | m210_01_operator p.9 (label 9) |  |  |  |
| component | processor_board | a component | no row | m410_01_dpc p.13 (label 13) |  |  |  |
| component | pump_oil_and_filter | a component | no row | m220_383_02_process_manual p.38 (label 38) |  |  |  |
| component | quickconnect | a component | no row | m330_01_gas_cabinet p.18 (label 18) |  |  |  |
| component | radiation_heat_shield | not extracted | child | m320_01_furnace p17 |  |  |  |
| component | safety_board | a component | no row | m410_01_dpc p.12 (label 12) |  |  |  |
| component | scr_module | parent furnace | parent power_supply | m320_01_furnace p.14 (label 14) |  |  |  |
| component | sic_paddle | parent furnace | parent load_station | m220_383_02_process_manual p.43 (label 43) |  |  |  |
| component | support_and_sealing | a component | no row | m320_01_furnace p.17 (label 17) |  |  |  |
| component | transformers | not extracted | child | m320_01_furnace p14 |  |  |  |
| component | tube_adapters | not extracted | child | m320_01_furnace p17 |  |  |  |
| component | vacuum_line | not extracted | child | m330_01_gas_cabinet p30 |  |  |  |
| component | wet_pump_oil | a component | no row | m220_383_02_process_manual p.36 (label 36) |  |  |  |
| fault_mode | flange: NH4Cl powder deposit on the flange | not extracted | yes | m220_383_02_process_manual p37 |  |  |  |
| fault_mode | heating_elements_condensate_buildup | a fault mode | no row | kanthal_fibrothal_heating p.13 |  |  |  |
| fault_mode | heating_elements_corrosion | a fault mode | no row | kanthal_fibrothal_heating p.13 |  |  |  |
| fault_mode | heating_elements_scaling | a fault mode | no row | kanthal_fibrothal_heating p.13 |  |  |  |
| fault_mode | inline_filter_clogging | a fault mode | no row | m330_01_gas_cabinet p.28 (label 28) |  |  |  |
| fault_mode | inline_gas_filters: filter clogging | not extracted | yes | m330_01_gas_cabinet p28 |  |  |  |
| fault_mode | mass_flow_controller_clogging | a fault mode | no (mass_flow_controllers: blockage by DCS condensation) | m220_383_02_process_manual p.37 (label 37) |  |  |  |
| fault_mode | mass_flow_controller_contamination | a fault mode | no row | mks_mfc_1179b p.75 (label 75) |  |  |  |
| fault_mode | mass_flow_controller_corrosion | a fault mode | no row | mks_mfc_1179b p.75 (label 75) |  |  |  |
| fault_mode | mass_flow_controller_swelling | a fault mode | no row | mks_mfc_1179b p.76 (label 76) |  |  |  |
| fault_mode | mass_flow_controllers: zero drift | not extracted | yes | mks_mfc_1179b p73 |  |  |  |
| fault_mode | o_rings_wear | a fault mode | no row | m320_01_furnace p.40 (label 40) |  |  |  |
| fault_mode | process_tube: deposition build-up on the quartz tube | not extracted | yes | m320_01_furnace p40 |  |  |  |
| fault_mode | process_tubes_deposit_buildup | a fault mode | no row | m220_383_02_process_manual p.38 (label 38) |  |  |  |
| fault_mode | process_tubes_dirt_contamination | a fault mode | no row | m220_383_02_process_manual p.44 (label 44) |  |  |  |
| fault_mode | pump_oil_and_filter_contamination | a fault mode | no row | m220_383_02_process_manual p.36 (label 36) |  |  |  |
| fault_mode | quick_connects_dirt_buildup | a fault mode | no row | m330_01_gas_cabinet p.18 (label 18) |  |  |  |
| fault_mode | quickconnect_corrosion | a fault mode | no row | m330_01_gas_cabinet p.18 (label 18) |  |  |  |
| fault_mode | sic_paddle_etching | a fault mode | no row | m320_01_furnace p.28 (label 28) |  |  |  |
| fault_mode | vacuum_line: deposit build-up in the vacuum line | not extracted | yes | m330_01_gas_cabinet p30 |  |  |  |
| fault_mode | vacuum_pump_system: vane wear | not extracted | yes | edwards_e2m_vacuum_pump p38 |  |  |  |
| fault_mode | wet_pump_oil_nh4cl_contamination | a fault mode | no row | m220_383_02_process_manual p.36 (label 36) |  |  |  |
