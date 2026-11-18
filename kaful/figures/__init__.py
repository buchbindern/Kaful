"""Figures drawn from saved artifacts only (M16). Not a pipeline phase: it reads every phase's artifacts.

Map:
- `spec`: the committed figure spec (lineages, figures, style);
- `sources`: opening a lineage by its evaluate run, recording every file read;
- `truth_view`: the only figures module that reads synthetic truth (through the scoring package);
- `forecasts`, `select`: sampled RUL and traces; the stated unit-selection rule;
- `f2_tracking`, `f3_rul_fan`, `f4_calibration`: one figure each (extract -> rows -> render);
- `style`, `drawn`: shared look, byte-stable encoding, display names;
- `build`, `commands`: `kaful figures build` and the figures manifest.
"""
