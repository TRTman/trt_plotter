# TRT Plotter — README

Small Python tool to plot TRT injections, hCG, PK (release-rate) and bloodwork (Total T, Free T, E2). The main script is `TRT_plot_and_sim_v2.py`. This README explains required CSV formats, configuration and how to run the plotter.

## Requirements
- Python 3.8+
- pip packages: pandas, numpy, matplotlib, python-dateutil, requests
- Install quickly:
    pip install pandas numpy matplotlib python-dateutil requests

## Running
- Edit configuration at the top of `TRT_plot_and_sim_v2.py` (dates, FIGSIZE, PK model, thresholds, sheet/CSV paths).
- Set `INJECT_SHEET` and `MARKER_SHEET` to either:
    - a Google Sheets edit URL (the script converts it to CSV), or
    - a local CSV file path.
- Run:
    python TRT_plot_and_sim_v2.py
- Output: `TRT_timeline_plot.png` and an interactive matplotlib window.

## Configuration highlights
- PLOT_START_DATE — plot window start (dd.mm.YYYY).
- BORON_START, PRE_BORON_START — shading/phases.
- TRT_PK_MODEL — `"elim"` (default) or `"abs"`.
- PK_FREQ — sampling frequency for PK curve (e.g. `'1h'`).
- ACTIVE_FRAC — fraction of dose considered "active" (tune as needed).
- E2_LOD_PG_ML and E2_PLOT_FRACTION_OF_LOD — how censored E2 ("<value") is plotted.
- HCG_ENABLED and HCG_PARAMS — model of hCG-driven endogenous T (half-life, delay, Emax, EC50).
- hCG detection threshold: `hcg_threshold` (default 124) used when parsing injections.

## CSV examples & example sheets
Below are compact example CSV snippets you can use as templates, followed by two example Google Sheets edit URLs you can point `INJECT_SHEET` and `MARKER_SHEET` at.

Injections CSV (rows: datetime, Substance, Amount mL, Amount mg or IU, inj site, comments):

```
datetime,Substance,Amount mL,Amount mg or IU,inj site,comments
08.04.2025 15:45,cyp 200mg/ml,0.35,70,right abdomen subq,"Example: 70 mg test cyp"
11.04.2025 08:05,,0.125,25,left abdomen subq,"Example: small microdose"
22.05.2025 13:45,HCG,0.09,250,left ab far outer,"Example: hCG dose (IU)"
```

Markers CSV (wide format: first column is marker name; following columns are sample dates):

```
Marker,2025-04-10,2025-04-15,2025-04-21,2025-05-13,2025-06-05
Total T (nmol/L),24.2,19.8,48.4,48.6,37.0
Total T (ng/dL),697.4,570.6,1394.7,1400.5,1066.2
SHBG (nmol/L),39,39,46,44,39
Albumin (g/L),45,44,43,42,43
Free T (pg/mL),137.0,109.2,304.5,320.0,238.0
Estradiol (pg/mL),<24,32.1,50.9,35.7,24.5
```

Example Google Sheets (edit URLs — the script converts these to CSV export links):

https://docs.google.com/spreadsheets/d/1sQGOocD6IM7zy2YPPrAED3C4V54HT3dAKlx0Wgih2yk/edit?usp=sharing

https://docs.google.com/spreadsheets/d/17WTx2jgGUAOwRhmfPhAlkkZNzy42INTAHLvDqI_gM2A/edit?usp=sharing

Use these as examples for the kinds of sheets you can point the script at. The rest of the README retains the format descriptions and tips; the inline example snippets that used to appear inside the format sections have been moved here to keep examples centralized.

## Injections CSV format
The script detects columns by substring matches. Required columns (case-insensitive substrings):
- A date/time column whose header contains `date` (e.g. `datetime`, `Date`, `date/time`). Values parsed by dateutil.parse — many formats accepted. Time is optional.
- A numeric column whose header contains `mg` (e.g. `Amount mg or IU`, `mg`) — this column is parsed for numeric dose values.
- Optional: a `Substance` column. If it contains `HCG` (case-insensitive) that row will be classified as hCG.

Notes:
- Rows classified as hCG are dropped from the TRT injection frame and used for hCG handling instead.
- The script also classifies a row as hCG if the numeric dose in the mg/IU column exceeds `hcg_threshold`.
- The loader is robust to minor formatting (it extracts digits from strings). Time-only or missing times are tolerated.

## Bloodwork (markers) CSV format
The marker loader expects a wide table: first column is the marker/test name; subsequent column headers are date/time strings (these headers are treated as sample timestamps). Each row represents a lab analyte.

- First column header can be anything (e.g. `Test`, `Analyte`) but each analyte row must include a recognizable name:
    - For Total testosterone rows use text containing `Total T` or `Total Testosterone`.
    - For Free testosterone rows use `Free T` or `Free Testosterone`.
    - For Estradiol use `E2` or `Estradiol`.
- Values are placed under date columns. Numeric values are expected. Censored values may be written with a leading `<` (e.g. `<24`) — the script will detect and plot those at a fraction of the LOD per config.

Notes:
- The loader searches the first-column strings to match the analyte name you pass to `load_markers(src, name)`. Provide `name` as `"Total T"`, `"Free T"` or `"Estradiol"`.
- Units should be consistent in a row (e.g. Total T in ng/dL is common). If you export both nmol/L and ng/dL as separate rows, the loader will select the row whose first-column text matches.

## Censored E2 handling
- Write censored results as `<value` (e.g. `<24`). Configure `E2_LOD_PG_ML` and the plotting fraction `E2_PLOT_FRACTION_OF_LOD` at the top of the script. The script will plot a point at the selected fraction of LOD and label it with the raw string.

## hCG detection & units
- If your `Substance` column contains `HCG` the row is treated as an hCG dose (IU).
- If no `Substance` column exists, large numeric values in the mg/IU column (above `hcg_threshold`) will be treated as hCG IU.

## Tips
- Use ISO-like or common date formats in column headers for the marker CSV (the loader parses headers as datetimes).
- If using Google Sheets, publish or make the sheet readable and pass the edit URL to the config variable — the script will fetch CSV export.
- Tune HCG_PARAMS and ACTIVE_FRAC to improve PK→TT calibration; the script contains both a simple fit and a more advanced Bateman+HCG calibration.