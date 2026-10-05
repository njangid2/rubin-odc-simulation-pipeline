# ODC Satellite Streak, Brightness & Pixel-Loss Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23171008.svg)](https://doi.org/10.5281/zenodo.23171008)

Simulates the impact of SpaceX Orbital Data Center (ODC) satellite constellations on
Vera C. Rubin Observatory (LSST) observations. The pipeline propagates a synthetic
94-shell satellite population through the Rubin pointing schedule, identifies
field-of-view crossings, models apparent brightness with a BRDF reflectance model,
converts that into per-band surface brightness, and finally estimates CCD pixel loss
(per-streak and per-pointing) from saturation, blooming, and dead-CCD effects.

Pixel loss is computed with **two approaches**: a **fast approach** (Steps 5, 7, 8) that
judges every streak on its own, and a **slow approach** (Step 9) that sums the electrons
of overlapping streaks and upgrades streaks that only cross the saturation threshold
because of that overlap. The paper's pixel-loss results use the slow approach unless
stated otherwise — see [Fast vs. Slow Pixel Loss](#fast-vs-slow-pixel-loss). A separate
step (Step 10) estimates the whole-sky brightness increase caused by the constellation.

Companion Jupyter notebooks (see [Notebooks](#notebooks)) turn the pipeline outputs into
the diagnostic figures and summary statistics used for analysis over a fixed one-year
survey window (`2026-06-29` → `2027-06-28`).

This is the simulation code for the paper *The Impact of Orbital Data Centers on the
NSF-DOE Vera C. Rubin Observatory* (Jangid, Rawls, Yoachim, Walke & Eggl). See
[Citing](#citing).

---

## Pipeline Overview

```
Step 1        Step 2         Step 3a/3b        Step 4          Step 4b
[Filter]  →  [Propagate]  →  [Az/El + Sun]  →  [Brightness]  →  [Band Correction]
                                                                       │
                                                                       ▼
                                                                  Step 5
                                                            [Pixel Loss v1]
                                                                       │
                                                                       ▼
                                                                  Step 6
                                                       [Per-Band Surface Brightness]
                                                                       │
                          ┌────────────────────────────────────────────┴───────────────┐
                          ▼                                                            ▼
              FAST pixel-loss approach                                  SLOW pixel-loss approach
              Step 7  [Pixel Loss v2, SB-corrected]                     Step 9  [Electron summing in
              Step 8  [CCD-level Combined Loss]                                 overlap regions]
                          │                                                            │
                          └────────────────────────────┬───────────────────────────────┘
                                                       ▼
                                      notebooks: plots1year, indetail_pxloss, ...
                                                [1-year analysis & figures]


Step 1 TLE cache  ──►  Step 10 [Whole-sky brightness increase]  ──►  nightsky_brightness.ipynb
```

Each step reads from the output directories of earlier steps. **Run steps in order.**
Steps 3a and 3b are independent of each other but both must finish before Step 4.
Step 5 is still required because Step 6 reads its per-streak output. After Step 6 the
pipeline splits into the **fast** approach (Steps 7–8) and the **slow** approach
(Step 9); you can run either or both. Step 10 (whole-sky brightness) only needs the
Step 1 TLE cache and `data_center_20deg.py`.

---

## Fast vs. Slow Pixel Loss

The fraction of focal-plane pixels lost to streaks is estimated in two ways. Both use the
same streak geometry, the same bright/normal split (`MAG_LIMIT`, 2.0 mag) and the same mask
widths (100 px for unsaturated, 500 px for saturated streaks). They differ in how a
*normal* streak is classified:

| | **Fast approach** (Steps 5, 7, 8) | **Slow approach** (Step 9) |
|---|---|---|
| Streak classification | Each streak is judged **alone**: its own peak electrons decide *unsaturated* / *saturated* / *faint* | The electrons of **all streaks crossing the same region are summed**; if the sum exceeds the full-well capacity (130,000 e⁻) that overlap sub-area is upgraded to *saturated* |
| Effect of overlaps | Overlapping streaks never make each other worse | Streaks that are individually unsaturated can become saturated (500 px mask) where they overlap |
| Bright streaks | Every CCD the streak touches is dead | Same |
| Total loss | Dead CCDs + union of the per-streak masks | Dead CCDs + union of masks rebuilt with the wider width where overlap upgrades occurred, clipped to the real CCD footprint (so the loss fraction is ≤ 100%) |
| Cost | Fast: one pass over the streaks | Slow: pixel-level electron accounting and a polygon-overlap search for every pointing; pointings with thousands of streaks dominate the run time |
| Use in the paper | Original method, kept for comparison | **Paper results** (unless stated otherwise) |

Step 9 computes both in one pass: its `old_*` output columns apply the fast approach's
logic (each streak judged alone) and its `new_*` columns give the slow approach, so the two
can be compared pointing by pointing (`delta_*`, `pct_increase`).

---

## Repository Layout

| Path | What it is |
|---|---|
| `Step1.py` … `step8_ccd_pixel.py` | Steps 1–8 (see [Step-by-Step Guide](#step-by-step-guide)) |
| `step9_updated_px_loss.py` | Step 9 — slow pixel loss (electron summing in overlap regions) |
| `1tenth_pixelloss.py` | Step 9 for a 1/10th-scale constellation |
| `step10_night_sky_brightness.py` | Step 10 — whole-sky brightness increase |
| `data_center_20deg.py` | Satellite brightness model imported by Steps 4 and 10 |
| `starlink/`, `analysis/` | Modules imported by `data_center_20deg.py` (`starlink.satellitemodels`, `analysis.calculator`) |
| `plots1year.ipynb` | One-year analysis figures and summary statistics → `plot_1year/` |
| `plotsky.ipynb` | Satellite sky maps → `sky_scatter_color_plots/` |
| `nightsky_brightness.ipynb` | Step 10 figures (e.g. `night_sky_brightness_*_all_bands_grid.png`) |
| `indetail_pxloss.ipynb` | Fast vs. slow classification for a single example pointing |

---

## Dependencies

```bash
pip install numpy pandas astropy sgp4 shapely lumos-sat matplotlib pytz healpy
```

You will also need:
- `data_center_20deg.py` — local satellite BRDF/surface model module, required by
  Step 4 (see dedicated section below). Must be in the same working directory as
  `step4_brightness_20deg.py`.
- `lumos-sat` — BRDF brightness framework used by Steps 4 and 6
  ([Fankhauser et al. 2023](https://github.com/Fankhauser/lumos)).
- `shapely` — polygon geometry used by Steps 5, 7, 8, and 9 for streak/CCD footprints.
- `rubin_sim` — imported by Step 8.
- `rubin_scheduler` (plus its data files) — only for Step 10 with `--sky-baseline real`,
  which uses `rubin_scheduler.skybrightness_pre.SkyModelPre` (the sky model of
  [RTN-012](https://rtn-012.lsst.io/)).
- `healpy` — healpix grid used by Step 10.
- `jupyter` — to open the notebooks.
- `starlink.satellitemodels` and `analysis.calculator` — modules that
  `data_center_20deg.py` imports for the base satellite surface model and the
  observer-frame intensity calculation. They are not on PyPI; they are provided as the
  `starlink/` and `analysis/` folders of this repository, so run Steps 4 and 10 from the
  repository root (or put the repository root on your `PYTHONPATH`).

> `data_center_20deg.py` is not pip-installable — keep it alongside the step scripts.

> `rubin_sim` and `rubin_scheduler` are licensed GPL-3.0; see [License](#license).

---

## Input Data

Before running Step 1 you need a Rubin scheduler (opsim) database, either as a
SQLite `.db` or a CSV export, containing at minimum:

| Column | Description |
|---|---|
| `observationId` | Unique pointing identifier |
| `fieldRA` | Right ascension of pointing center (deg) |
| `fieldDec` | Declination of pointing center (deg) |
| `observationStartMJD` | Exposure start time (MJD) |
| `visitExposureTime` | Exposure duration (seconds) |
| `filter` | Photometric filter (raw opsim value, e.g. `r_57`) |
| `night` | Observing night index |
| `rotSkyPos` | Camera rotation on sky, East of North (needed for Step 8 only) |

Pointing data can be obtained from the
[Rubin baseline scheduler database](https://rubin-scheduler.lsst.io).

---

## ⚠️ Known Data Quirk — `pointing_filter` is suffixed

The raw opsim `filter` column (carried through as `pointing_filter` in every step2+
output) is **not** a bare band letter. It is suffixed with a filter-load /
throughput-curve id, e.g. `"r_57"`, `"g_12"`. Any code that looks up a per-band
constant (solar color, zero-point, throughput) must first strip the suffix:

```python
base_band = str(pointing_filter).split("_")[0]   # "r_57" -> "r"
```

Steps 4b, 5, 7, and 8 each apply this fix independently (look for `base_band()` /
`str.split("_")[0]` in each script). If you fork or extend the pipeline, apply the
same fix anywhere you index into a band-keyed dictionary (`u`/`g`/`r`/`i`/`z`/`y`) —
otherwise every lookup silently misses and the affected column comes back all-NaN
(or every streak gets misclassified as "faint").

---

## Step-by-Step Guide

### Step 1 — Affected Exposure Identification
**Script:** `Step1.py`
**Output directory:** `streak_output_sgp4/`

Identifies which Rubin pointings are plausibly affected by each constellation shell.
For each shell and each calendar day, one representative satellite per orbital plane
is propagated across the full day using SGP4. A pointing is flagged if any
representative satellite passes within `match_deg` (default 5°) of the pointing
center at any sampled time. This is a fast recall-favoring prefilter — some flagged
pointings will have no true crossing, but genuine crossings are not missed.

Also generates and caches per-shell, per-day TLE `.pkl` files used by Step 2.

**Outputs:**
- `streak_output_sgp4/streak_results_shell_XXX.csv` — one file per shell, pointings with `n_streaks > 0`
- `streak_output_sgp4/tles_shell_XXX_YYYY-MM-DD.pkl` — synthetic TLE cache
- `streak_output_sgp4/step1.log`

```bash
python Step1.py
```

---

### Step 2 — Propagation and Field-of-View Matching
**Script:** `step2.py`
**Input:** `streak_output_sgp4/`
**Output directory:** `step2_output/`

For every pointing flagged in Step 1, propagates all satellites in that shell across
the full exposure window (10 time steps per exposure, ~3 s resolution for a 30 s
exposure). Satellite positions are transformed from TEME to RA/Dec and matched
against the Rubin field-of-view radius (1.75°). Also computes a sunlit/shadow flag
per satellite per time step using a cylindrical Earth-shadow model.

**Outputs:**
- `step2_output/streak_trajectories_YYYY-MM-DD.csv` — one file per day, columns:
  `pointing_id`, `pointing_ra`, `pointing_dec`, `pointing_mjd`, `pointing_exptime`,
  `pointing_filter`, `pointing_night`, `shell_id`, `sat_name`, `step`, `t_mjd`,
  `ra_deg`, `dec_deg`, `sep_fov_center_deg`, `in_fov`, `sunlit`
- `step2_output/step2.log`

```bash
python step2.py
```

**Resumability:** two levels — a completed day CSV skips the whole day; a completed
per-shell intermediate CSV skips just that shell within a day.

---

### Step 3a — Satellite Azimuth/Elevation
**Script:** `step3a.py`
**Input:** `step2_output/`
**Output directory:** `step3_output/`

Transforms satellite RA/Dec from Step 2 into observer-frame azimuth/elevation at
the Rubin site. Computed only for sunlit + in-FOV rows (fast); all other rows are
NaN. Output is strictly row-aligned with the matching step2 CSV — no join needed.

**Outputs:**
- `step3_output/streak_trajectories_YYYY-MM-DD_azel.csv` — columns: `az_deg`, `el_deg`

```bash
python step3a.py
```

---

### Step 3b — Sun Position
**Script:** `step3b.py`
**Input:** `step2_output/`
**Output directory:** `step3_output/`

Computes Sun azimuth/elevation at the Rubin site for every **unique** `t_mjd` value
across all step2 files, once, and stores it as a lookup table — far cheaper than
recomputing per row or per day since the Sun position only depends on time.

**Outputs:**
- `step3_output/sun_positions.csv` — columns: `t_mjd`, `sun_az_deg`, `sun_el_deg`

```bash
python step3b.py
```

> Step 3a and Step 3b are independent and can run in either order, but both must
> finish before Step 4.

---

### Step 4 — Brightness Modeling
**Script:** `step4_brightness_20deg.py`
**Requires:** `data_center_20deg.py` in the same directory
**Input:** `step2_output/`, `step3_output/`
**Output directory:** `step4_output_20deg/`

Computes apparent AB magnitude (at ~532 nm) for every sunlit, in-field satellite
crossing, from the instantaneous Sun–satellite–observer geometry using a
BRDF-based reflectance model (Lumos-Sat). Satellite model:
- **Chassis:** Starlink V2 Mini BRDF scaled to ~7 × 3.5 m²
- **Solar array:** Starlink V1.5 BRDF scaled to 400 kW power (1,679 m² panel area)
- **Panel offset:** 20° from Sun toward nadir (Rodrigues rotation)

Brightness is `NaN` for rows that are shadowed, out of FOV, or missing geometry.
AB magnitude is clipped at roughly 12 mag for unphysical low-intensity cases.
Earthshine is not modeled — only direct solar illumination.

**Outputs:**
- `step4_output_20deg/streak_trajectories_YYYY-MM-DD_bright.csv` — column: `ab_magnitude`

```bash
python step4_brightness_20deg.py
```

**Resumability:** skips a day if the output row count matches the step2 input row count.

---

### Step 4b — Per-Band Magnitude Correction
**Script:** `step4b_convertopbandmag.py`
**Input:** `step2_output/` (for `pointing_filter`), `step4_output_20deg/` (for `ab_magnitude`)
**Output directory:** `step4b_output/`

Step 4's `ab_magnitude` assumes a single ~532 nm passband regardless of which LSST
filter (`u`/`g`/`r`/`i`/`z`/`y`) the pointing actually used. Step 4b applies a
per-row solar-color offset (Willmer 2018 / SMTN-002) to shift the 532 nm magnitude
into the pointing's real band:

```
ab_magnitude_corrected = ab_magnitude_532nm + solar_color[base_band(pointing_filter)]

u: +1.428   g: +0.245   r: -0.210   i: -0.322   z: -0.357   y: -0.371
```

**Outputs:**
- `step4b_output/streak_trajectories_YYYY-MM-DD_bright_corrected.csv` — columns:
  `pointing_filter`, `solar_color_applied`, `ab_magnitude_corrected`

```bash
python step4b_convertopbandmag.py
```

---

### Supporting Module — `data_center_20deg.py`

Not a pipeline step itself — this is the satellite brightness calculator imported
by Step 4 (`import data_center_20deg as data_center`). It builds the satellite
surface/BRDF model and converts observer-frame geometry into intensity and AB
magnitude.

**Satellite model:**
- Base satellite surfaces from `starlink.satellitemodels.get_surfaces()`
  (chassis + bus surfaces), plus:
- A solar array surface sized from `power_kw` via `power_to_area()`
  (`area = (power_kw / 25.0 kW) * 104.96 m²`, doubled if `continuous=True`),
  using a `BINOMIAL` BRDF fit for the array material.
- Two radiator surfaces (110 m² each, Lambertian, albedo 0.9), nominally normal
  to the body x-axis with a small mechanical wobble (default ±5°) applied around
  the y or z axis.
- An Earth BRDF (`PHONG(Kd=0.2, Ks=0.2, n=300)`) used only if `include_earthshine=True`.

**Solar panel off-pointing (brightness mitigation):**
The solar array's surface normal does *not* point exactly at the Sun — it is
rotated `PANEL_OFFSET_DEG` (default 20°) from the Sun direction toward nadir,
via `calculate_panel_normal()` using the Rodrigues rotation formula around the
axis `k = sun_vector × nadir_vector`. This models satellites deliberately
tilting their panels away from the Sun to reduce reflected brightness:
- `offset_deg = 0` → panel faces the Sun exactly (maximum brightness)
- `offset_deg = 20` → panel receives `cos(20°) ≈ 94%` of peak solar flux (this repo's default)
- `offset_deg = 90` → panel faces nadir, no direct solar reflection

**Key functions:**
| Function | Purpose |
|---|---|
| `calculate_sun_direction_vectors(sun_alt, sun_azi)` | Sun az/el → unit vector (x=East, y=North, z=Up) |
| `calculate_panel_normal(sun_alt, sun_azi, offset_deg)` | Offset panel normal via Rodrigues rotation |
| `validate_panel_offset(...)` | Sanity-checks `dot(panel_normal, sun_dir) ≈ cos(offset_deg)` |
| `power_to_area(power_kw, continuous)` | Solar power (kW) → panel area (m²) |
| `get_surfaces_with_solar_array(power_kw, sun_altitude, sun_azimuth, ...)` | Assembles chassis + tilted solar array into a `Surface` list |
| `surface_with_radiators(surfaces, wobble_deg, wobble_axis)` | Appends the two wobbled radiator surfaces |
| `calculate_brightness(sat_height, sat_altitude, sat_azimuth, sun_altitude, sun_azimuth, power_kw, ...)` | Full pipeline: builds surfaces → `calculator.get_intensity_observer_frame()` → `intensity_to_ab_mag()`. Returns a dict with `intensity`, `ab_magnitude`, `area`, `power_type`, `sun_normal`, `offset_deg`, `dot_panel_sun` |
| `intensity_to_ab_mag(intensity, clip=True)` | Converts W/m² intensity to AB magnitude at the 532 nm reference wavelength, clipping below ~12 mag if `clip=True` |

All function signatures/return values match the original (pre-offset) version of
this module, so Step 4 and any other caller require no changes when swapping
`PANEL_OFFSET_DEG`.

---

### Step 5 — Pixel Loss (v1, fast approach)
**Script:** `step5_streak_pixelloss.py`
**Input:** `step2_output/`, `step3_output/`, `step4_output_20deg/`
**Output directory:** `step5_output/`

Computes streak brightness impact and CCD pixel loss for every pointing. Reads
`pointing_filter` from step2 and immediately reduces it to the bare base-band
letter (`pointing_filter_raw` keeps the original suffixed value for traceability),
so every downstream script can treat it as a plain single-letter band.

**Outputs (per day, plus one running summary):**
- `step5_output/step5_datapoints_YYYY-MM-DD.csv` — per streak (pointing × sat × shell):
  `pointing_id`, `sat_name`, `shell_id`, `pointing_filter`, `pointing_filter_raw`,
  `pointing_exptime`, `ab_magnitude`, `streak_type`, `ang_vel_arcsec_s`, `L_px`,
  `peak_electrons`, `sat_status`, `pixel_loss_psf`, `pixel_loss_blooming`,
  `pixel_loss_saturated`, `pixel_loss_unsaturated`, `pixel_loss_faint`,
  `pixel_loss_recommended`
- `step5_output/step5_pointings_YYYY-MM-DD.csv` — per pointing: `pointing_id`,
  `pointing_ra`, `pointing_dec`, `pointing_filter`, `pointing_filter_raw`,
  `pointing_exptime`, `n_streaks`, `total_pixel_loss_psf_fraction`,
  `total_pixel_loss_recommended_fraction`, `n_blooming`, `n_saturated`,
  `n_unsaturated`, `n_faint`
- `step5_output/step5_daily_summary.csv` — per day (appended): `date`,
  `pointing_night`, `n_pointings`, `n_affected_pointings`,
  `mean/max_pixel_loss_psf_fraction`, `mean/max_pixel_loss_recommended_fraction`,
  `total_blooming/saturated/unsaturated/faint`

```bash
python step5_streak_pixelloss.py
```

---

### Step 6 — Per-Band Surface Brightness
> Used by **both** pixel-loss approaches (Steps 7/8 and Step 9).

**Script:** `step6_Streak_perband.py`
**Input:** step5 `step5_datapoints_*.csv` + `step2_output/` + `step3_output/`
**Output directory:** `step6_output_perband/`

Computes streak surface brightness (mag/arcsec²) per day, per LSST band, using the
correct per-band zero-point, throughput, and solar-color correction (SMTN-002 /
PSTN-054 / `syseng_throughputs` v1.9), plus slant-range defocus and PSF widening.

> **Note:** the script's `step5_dir` default (`"step8_output_pixel_loss"`) is a
> historical folder name from earlier pipeline iterations — point it at wherever
> Step 5's `output_dir` actually landed (e.g. `step5_output`) when you call `main()`.

**Outputs:**
- `step6_output_perband/step6_YYYY-MM-DD_{band}.csv` — one file per day per band,
  columns include: `pointing_id`, `sat_name`, `shell_id`, `pointing_filter`,
  `pointing_exptime`, `ab_magnitude`, `ab_magnitude_band`, `solar_color_applied`,
  `detectable`, `L_px`, `shell_alt_km`, `ang_vel_arcsec_per_s`, `t_crossing_sec`,
  `median_elevation_deg`, `theta_defocus_arcsec`, `theta_eff_arcsec`,
  `peak_electrons`, `streak_brightness_e_per_px`, `surface_brightness_mag_arcsec2`

```bash
python step6_Streak_perband.py
```

**Parallelism note:** days are processed simultaneously (outer `Pool`); bands are
processed sequentially within each day worker to avoid nested pools.

---

### Step 7 — Pixel Loss (v2, Surface-Brightness Corrected; fast approach)
**Script:** `step7_streak_corrected_after_loss.py`
**Input:** `step2_output/` (geometry, `L_px`), `step6_output_perband/` (surface brightness)
**Output directory:** `step8_output_pixel_loss_v2/`

Recomputes pixel loss like Step 5, but reads `surface_brightness_mag_arcsec2`
directly from Step 6's output instead of re-deriving it from `ab_magnitude` with a
simplified single-band formula. This removes Step 5 v1's incorrect throughput/solar
color handling and its dependency on the Step 4 bright files. Also carries the
`pointing_filter` suffix fix (see the quirk note above) — without it, every streak
was previously misclassified as "faint" and saturated/unsaturated counts were
always zero.

**Outputs:** same schema/naming pattern as Step 5:
- `step8_output_pixel_loss_v2/step5_datapoints_YYYY-MM-DD.csv`
- `step8_output_pixel_loss_v2/step5_pointings_YYYY-MM-DD.csv`
- `step8_output_pixel_loss_v2/step5_daily_summary.csv`

```bash
python step7_streak_corrected_after_loss.py
```

---

### Step 8 — CCD-Level Combined Pixel Loss (fast approach)
**Script:** `step8_ccd_pixel.py`
**Input:** `step2_output/`, `step4_output_20deg/`, `step6_output_perband/`, opsim DB (for `rotSkyPos`)
**Output directory:** `step5_combined_loss_output_mag7/`

Computes combined pixel loss for pointings containing at least one streak brighter
than `MAG_LIMIT`:

- **Bright streaks** (`ab_magnitude < MAG_LIMIT`): every CCD the streak touches is
  marked entirely dead.
- **Normal streaks** (`ab_magnitude >= MAG_LIMIT`): classified saturated /
  unsaturated / faint from Step 6 surface brightness, and contribute a Shapely
  streak polygon (with dead CCDs subtracted) to the total loss.

Streak paths are projected to focal-plane pixels using a gnomonic projection
corrected for each pointing's `RotSkyPos` (camera rotation East of North, per
SMTN-019), including the 180° flip from the odd number of mirrors in the Simonyi
Survey Telescope:

```
angle = 180° - RotSkyPos
x_cam =  cos(angle)*x_sky + sin(angle)*y_sky
y_cam = -sin(angle)*x_sky + cos(angle)*y_sky
```

**Outputs:**
- `step5_combined_loss_output_mag7/combined_datapoints_YYYY-MM-DD.csv`
- `step5_combined_loss_output_mag7/combined_pointings_YYYY-MM-DD.csv`
- `step5_combined_loss_output_mag7/combined_daily_summary.csv`

```bash
python step8_ccd_pixel.py
```

> Step 8 judges every streak on its own; overlapping streaks are not combined. For that,
> see Step 9.

---

### Step 9 — Pixel Loss, Slow Approach (Electron Summing in Overlap Regions)
**Script:** `step9_updated_px_loss.py`
**Input:** `step2_output/`, `step4b_output_rad/`, `step6_output_perband_rad/`, opsim DB (for `rotSkyPos`)
**Output directory:** `step9_output/`

Runs the slow pixel-loss approach over **every pointing of every date** in the window. The
`_rad` input directories are the Step 4b / Step 6 outputs made with the radiator-inclusive
satellite model; pass `--step4b-dir` / `--step6-dir` to use others.

- **Bright streaks** (band-corrected `ab_magnitude_corrected` < `MAG_LIMIT`, 2.0 mag):
  every CCD the streak touches is entirely dead.
- **Normal streaks:** peak electrons come from the Step 6 surface brightness (not the raw
  magnitude), projected with the pointing's real camera rotation (`rotSkyPos`). Each streak
  gets a mask polygon along its path, `MASK_WIDTH_PX` wide: 100 px if unsaturated, 500 px
  if saturated.
- **Overlaps:** where two streaks' mask polygons overlap, their electrons are summed for
  *that overlap region only*. If the sum exceeds the full-well capacity (130,000 e⁻), only
  that sub-area is upgraded to saturated — not the whole streak.
- **Totals:** dead CCDs and normal-streak masks are combined into one union, clipped to the
  real CCD footprint, and divided by the footprint area, so the loss fraction never
  exceeds 100%.

Dates are processed in parallel (`--workers`, default 12; each worker holds one date in
memory — 20 workers once triggered an out-of-memory kill in a 200 GB job) and the whole
simulation can be split across Slurm array tasks with `--num-shards` / `--shard-index`
(each shard gets a disjoint, round-robin subset of dates, so shards never write to the
same file). Only dates in `START_DATE`–`END_DATE` (`2026-06-29` → `2027-06-28`, inclusive)
are processed; edit those two constants in the script to change the window.

**Outputs:** one CSV per date, `step9_output/pixel_loss_summary_YYYY-MM-DD.csv`, one row
per pointing:

| Column(s) | Meaning |
|---|---|
| `date`, `pointing_id`, `pointing_filter`, `rot_sky_pos` | Identifiers and the camera rotation used |
| `n_bright_streaks`, `n_normal_streaks` | Number of bright (dead-CCD) and normal streaks |
| `n_dead_ccds`, `dead_pixel_loss_px2` | CCDs killed by bright streaks, and their area (px²) |
| `old_normal_area_px2`, `new_normal_area_px2` | Normal-streak mask area (px²): fast vs. slow |
| `old_total_px2`, `new_total_px2` | Total lost area (px²): fast vs. slow |
| `old_total_fraction`, `new_total_fraction` | The same, as a fraction of the focal-plane footprint: **fast** vs. **slow** (use `new_*` for the paper's numbers) |
| `delta_px2`, `delta_fraction`, `pct_increase` | Slow minus fast (area, fraction), and the relative increase in % |
| `n_upgraded_centerline` | Streaks upgraded when electrons are summed along rasterized centerlines |
| `n_upgraded_mask` | Streaks upgraded using the mask-polygon overlap (the realistic version) |
| `n_overlap_pairs` | Number of intersecting mask-polygon pairs |
| `status`, `error` | `ok`, or the error that occurred for that pointing |

```bash
# whole simulation with defaults
python step9_updated_px_loss.py

# quick check on two dates, single worker
python step9_updated_px_loss.py --max-dates 2 --workers 1 --verbose

# diagnostic 3-panel plots for the 20 pointings with the largest fast -> slow change
python step9_updated_px_loss.py --plot-top-n 20 --plot-dir step9_output/plots
```

Example Slurm job array (7 shards; adjust the resources to your cluster):

```bash
#!/bin/bash
#SBATCH --array=0-6
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=6
#SBATCH --mem=300G
#SBATCH --time=6-20:00:00
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=step9_shard_%a.log

python3 step9_updated_px_loss.py \
    --workers 6 \
    --num-shards 7 \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --out-dir step9_output
```

**Key parameters** (constants near the top of the script):

| Constant | Value | Meaning |
|---|---|---|
| `MAG_LIMIT` | 2.0 mag | Streaks brighter than this are "bright": the whole CCD is dead |
| `MASK_WIDTH_PX` | 100 / 500 px | Mask width of unsaturated / saturated normal streaks |
| `FULL_WELL_E` | 130,000 e⁻ | LSSTCam full-well capacity (saturation threshold) |
| `PSF_PEAK_FRAC` | set in the script | Fixed fraction of a streak's flux assumed to fall in its peak pixel. This is an assumed constant; see the limitation discussed in Section 2.7.1 of the paper |
| `START_DATE`, `END_DATE` | 2026-06-29, 2027-06-28 | Date window (inclusive) |

---

### Step 9 variant — 1/10th-Scale Constellation
**Script:** `1tenth_pixelloss.py`
**Input:** same as Step 9
**Output directory:** `tenth_constellation_step9_output/`

Simulates a constellation with one tenth of the satellites. It is Step 9 with exactly one
addition: before any per-pointing physics runs, `subsample_satellites()` keeps only every
10th satellite within each (shell, plane) — satellites 1, 11, 21, … (names are
`SH{shell}-{plane}-{sat}`, so the plane is read from the name). The same satellites are
kept on every date. Physics, output format (`pixel_loss_summary_YYYY-MM-DD.csv`), sharding
and resumability are otherwise identical to Step 9; the date window is set by `FIRST_DATE`
and `CUTOFF_DATE` (exclusive) near the top of the script.

```bash
python 1tenth_pixelloss.py --workers 6 --num-shards 7 --shard-index "$SLURM_ARRAY_TASK_ID" \
    --out-dir tenth_constellation_step9_output
```

---

### Step 10 — Whole-Sky Night Sky Brightness
**Script:** `step10_night_sky_brightness.py`
**Requires:** `data_center_20deg.py` (and `starlink/`, `analysis/`) in the working directory
**Input:** `streak_output_sgp4/tles_shell{id}_{date}.pkl` (the Step 1 TLE cache)
**Output:** `night_sky_brightness_YYYY-MM-DD.csv`

A different quantity from pixel loss: pixel loss concerns streaks crossing the camera's
narrow field of view during exposures, whereas Step 10 computes the **diffuse brightness
added to the whole sky** by every satellite that is above the horizon and sunlit at a
given moment, whether or not a telescope is looking at it. For a chosen night
(astronomical night, Sun elevation ≤ −18°) and time step it:

1. propagates every satellite of every shell with SGP4 and determines which are sunlit
   (cylindrical Earth-shadow model, same as Step 2) and above the elevation cutoff;
2. computes each satellite's brightness with `data_center_20deg.calculate_brightness()`
   (same model and parameters as Step 4);
3. sums the flux of all satellites, `F_total = Σ 10^(−0.4·m_i)`, and converts it to an
   added surface brightness over the sky region;
4. combines it with a natural-sky baseline to give the magnitude and percentage increase.

The natural-sky baseline (`--sky-baseline`) is one of: `fixed` (a constant, default
21.8 mag/arcsec²), `real` (per-timestep, per-band sky brightness from
`rubin_scheduler.skybrightness_pre.SkyModelPre`, [RTN-012](https://rtn-012.lsst.io/) —
it varies with Moon phase and twilight), or `opsim` (per-visit sky brightness read from an
opsim database).

```bash
python step10_night_sky_brightness.py --date 2026-06-29
python step10_night_sky_brightness.py --date 2026-06-29 --sky-baseline real --band r
python step10_night_sky_brightness.py --date 2026-06-29 --elevation-cutoff-deg 20 --time-step-sec 300
```

---

## Notebooks

### `plots1year.ipynb` — one-year analysis

Consumes the outputs of Steps 2, 3, 4, 5, 6, 7, and 8 to produce the diagnostic
figures and summary statistics used for a fixed 365-day analysis window
(`2026-06-29` → `2027-06-28`). Includes, among others:

- Stacked brightness histogram of per-streak minimum AB magnitude, by shell
- Per-pointing brightness scatter (mean / brightest)
- Twilight-restricted pointing plots (sun elevation 0° to −25°)
- Pixel-loss heatmap and pixel-loss-over-time plots (merged combined-loss + v2 sources)
- Per-band, per-pointing surface brightness plots
- Pixel-loss distribution ("how often / how much") histograms
- Average streaks-per-day (total vs. sunlit-only)
- Brightest individual streak by surface brightness, across all six bands
- Survey-wide summary statistics (twilight fraction, max/mean pixel loss, etc.)
- Coverage diagnostics (missing days / rows vs. expected pointings for the window)

Open with:

```bash
jupyter notebook plots1year.ipynb
```

Figures are written to `plot_1year/`.

### `plotsky.ipynb` — satellite sky maps

Azimuth/elevation sky maps of the constellation at astronomical twilight (Sun elevation
≈ −21°) for three dates, coloured by orbital group (LEO-30, H-LEO-30, SSO-97, SSO-99;
satellites in Earth's shadow in gray), with the Rubin elevation limits (15°–86.5°) and
per-panel satellite counts. It works for the full constellation or the 1/10th-scale subset
(every 10th satellite per plane) and reads the Step 1 TLE cache. Figures are written to
`sky_scatter_color_plots/`.

### `nightsky_brightness.ipynb` — whole-sky brightness (Step 10)

Plots the Step 10 time series: natural vs. natural + constellation sky brightness and the
percentage increase, for all six bands. Example figures for a full-moon night (2026-06-29,
Moon ≈ 99% illuminated) and a new-moon night (2026-07-15, ≈ 1%) are
`night_sky_brightness_2026-06-29_all_bands_grid.png` and
`night_sky_brightness_2026-07-15_all_bands_grid.png`.

### `indetail_pxloss.ipynb` — fast vs. slow for a single pointing

An in-detail look at what the slow approach does, for one pointing (default:
`pointing_id` 23838 on 2026-07-31, a high-pixel-loss example with a total pixel loss of
about 51.5%). It draws two panels with the functions of `step9_updated_px_loss.py`:

- **NAIVE** (left): every normal streak judged alone (the fast approach's classification).
- **COMBINED** (right): overlapping streaks' electrons summed along their centerlines;
  streaks upgraded by the summation (e.g. unsaturated → saturated) are drawn in black.

Line widths equal the real mask widths (100 px unsaturated, 500 px saturated) and ultrabright
streaks are red. It needs `step9_updated_px_loss.py` in the same directory and the Step 2 /
4b / 6 outputs for that date. Usage: `run("2026-07-31", 23838)`; the figure is saved to
`plots_naive_vs_combined/`.

---

## Joining Pipeline Outputs

To assemble a full per-row table for one day:

```python
import pandas as pd

date = "2025-11-01"
df       = pd.read_csv(f"step2_output/streak_trajectories_{date}.csv")
azel     = pd.read_csv(f"step3_output/streak_trajectories_{date}_azel.csv")
sun      = pd.read_csv("step3_output/sun_positions.csv").set_index("t_mjd")
bright   = pd.read_csv(f"step4_output_20deg/streak_trajectories_{date}_bright.csv")
band_mag = pd.read_csv(f"step4b_output/streak_trajectories_{date}_bright_corrected.csv")

df["az_deg"]                 = azel["az_deg"]
df["el_deg"]                 = azel["el_deg"]
df["sun_az_deg"]             = df["t_mjd"].map(sun["sun_az_deg"])
df["sun_el_deg"]             = df["t_mjd"].map(sun["sun_el_deg"])
df["ab_magnitude"]           = bright["ab_magnitude"]
df["ab_magnitude_corrected"] = band_mag["ab_magnitude_corrected"]
```

Per-streak pixel loss and surface brightness (`step5_output/`,
`step6_output_perband/`, `step8_output_pixel_loss_v2/`) and CCD-level combined loss
(`step5_combined_loss_output_mag7/`) join on `pointing_id` / `sat_name` /
`shell_id`, not by row position — see each step's section above for exact columns.

Step 9 writes one CSV per date; to load all of them:

```python
import glob
import pandas as pd

step9 = pd.concat(
    (pd.read_csv(p) for p in sorted(glob.glob("step9_output/pixel_loss_summary_*.csv"))),
    ignore_index=True,
)
step9 = step9[step9["status"] == "ok"]

step9["new_total_fraction"]   # slow approach (electron summing) -- paper results
step9["old_total_fraction"]   # fast approach (each streak judged alone)
```

---

## Resumability

All steps are resume-safe. If a run is interrupted, just re-run the same script:

- **Step 1:** skips shells whose output CSV already exists
- **Step 2:** skips days whose merged CSV exists; skips individual shells within a day
- **Step 3a/3b:** skips dates/files already computed
- **Step 4:** skips a day when the output row count matches the step2 input row count
- **Step 4b–8:** skip any per-day output file that already exists
- **Step 9 / 1/10th:** skip every date whose `pixel_loss_summary_*.csv` already exists and is
  verified complete; a date interrupted mid-write is redone. A single bad pointing is
  recorded as an error row and does not stop the run
- **Step 10:** one output CSV per night; re-running a night overwrites it

---

## Constellation Shell Reference

The pipeline models 94 ODC shells across four orbital groups:

| Group | Shell IDs | Altitude (km) | Inclination |
|---|---|---|---|
| LEO-30 | 1–25 | 686–718 | ~30° |
| H-LEO-30 | 26–50 | 946–978 | ~30° |
| SSO-97 | 51–72 | 707–744 | ~97–98° |
| SSO-99 | 73–94 | 967–1002 | ~99.4–99.5° |

---

## Observatory Location

All observer-frame calculations use the Vera C. Rubin Observatory site:

| Parameter | Value |
|---|---|
| Latitude | 30° 14′ 40.70″ S |
| Longitude | 70° 44′ 57.90″ W |
| Elevation | 2663 m |

---

## Notes

- All steps support parallel processing via `multiprocessing.Pool`; worker counts
  are set in each script's `main()` call and should be tuned to available cores.
- TLE `.pkl` files from Step 1 can be deleted after Step 2 by setting
  `delete_tles=True` in `step2.py`.
- The brightness model (Step 4) clips AB magnitude at ~12 mag for unphysical
  low-intensity cases.
- Earthshine is not modeled anywhere in the pipeline — only direct solar
  illumination is considered.
- See the **Known Data Quirk** section above before extending any script that
  reads `pointing_filter`.
- Step 9 and `1tenth_pixelloss.py` parallelize across **dates** with a process pool; keep
  `--workers` conservative (default 12) because each worker loads a whole date into memory.
- `PSF_PEAK_FRAC` (Step 9) is a fixed, assumed constant applied to the defocus-broadened
  streak density; see the limitation noted in Section 2.7.1 of the paper.

---

## Citing

If you use this software, please cite the archived release:

> Jangid, N. (2026). *ODC Simulation Pipeline: satellite-streak, pixel-loss and
> sky-brightness simulations for orbital data centers observed by Vera C. Rubin
> Observatory* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23171008

and the associated paper (Jangid, Rawls, Yoachim, Walke & Eggl, *The Impact of Orbital Data
Centers on the NSF-DOE Vera C. Rubin Observatory*). GitHub's "Cite this repository" button
reads [`CITATION.cff`](CITATION.cff).

---

## License

This software is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version (**GPL-3.0-or-later**).
It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
[`LICENSE`](LICENSE) file for the full text.

The GPL license is used because Step 8 imports `rubin_sim` and Step 10 imports
`rubin_scheduler`, which are both licensed GPL-3.0.

---

## Authorship and Acknowledgements

All coding work was done by **Nayan Jangid**, with some assistance from Claude
(Anthropic's AI assistant) for debugging, performance improvements, and parts of the
plotting and analysis scripts.
