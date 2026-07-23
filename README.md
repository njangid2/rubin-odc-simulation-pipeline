# ODC Satellite Streak, Brightness & Pixel-Loss Pipeline

Simulates the impact of SpaceX Orbital Data Center (ODC) satellite constellations on
Vera C. Rubin Observatory (LSST) observations. The pipeline propagates a synthetic
94-shell satellite population through the Rubin pointing schedule, identifies
field-of-view crossings, models apparent brightness with a BRDF reflectance model,
converts that into per-band surface brightness, and finally estimates CCD pixel loss
(per-streak and per-pointing) from saturation, blooming, and dead-CCD effects.

A companion Jupyter notebook (`plots1year.ipynb`) turns the pipeline outputs into the
diagnostic figures and summary statistics used for analysis over a fixed one-year
survey window.

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
                                                                       ▼
                                                                  Step 7
                                                       [Pixel Loss v2, SB-corrected]
                                                                       │
                                                                       ▼
                                                                  Step 8
                                                        [CCD-level Combined Loss]
                                                                       │
                                                                       ▼
                                                            plots1year.ipynb
                                                         [1-year analysis & figures]
```

Each step reads from the output directories of earlier steps. **Run steps in order.**
Steps 3a and 3b are independent of each other but both must finish before Step 4.

---

## Dependencies

```bash
pip install numpy pandas astropy sgp4 shapely lumos-sat matplotlib pytz
```

You will also need:
- `data_center_20deg.py` — local satellite BRDF/surface model module, required by
  Step 4 (see dedicated section below). Must be in the same working directory as
  `step4_brightness_20deg.py`.
- `lumos-sat` — BRDF brightness framework used by Steps 4 and 6
  ([Fankhauser et al. 2023](https://github.com/Fankhauser/lumos)).
- `shapely` — polygon geometry used by Steps 5, 7, and 8 for streak/CCD footprints.
- `starlink.satellitemodels` and `analysis.calculator` — internal packages that
  `data_center_20deg.py` imports for the base satellite surface model and the
  observer-frame intensity calculation. These are not on PyPI; they must be on
  your `PYTHONPATH` (from your organization's internal `lumos`/`starlink`
  toolchain) before Step 4 will import successfully.

> `data_center_20deg.py` is not pip-installable — keep it alongside the step scripts.

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

### Step 5 — Pixel Loss (v1)
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

### Step 7 — Pixel Loss (v2, Surface-Brightness Corrected)
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

### Step 8 — CCD-Level Combined Pixel Loss
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

---

## Analysis Notebook — `plots1year.ipynb`

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

---

## Resumability

All steps are resume-safe. If a run is interrupted, just re-run the same script:

- **Step 1:** skips shells whose output CSV already exists
- **Step 2:** skips days whose merged CSV exists; skips individual shells within a day
- **Step 3a/3b:** skips dates/files already computed
- **Step 4:** skips a day when the output row count matches the step2 input row count
- **Step 4b–8:** skip any per-day output file that already exists

---

## Constellation Shell Reference

The pipeline models 94 ODC shells across four orbital groups:

| Group | Shell IDs | Altitude (km) | Inclination |
|---|---|---|---|
| LEO-30 | 1–25 | 686–718 | ~30° |
| MEO-30 | 26–50 | 946–978 | ~30° |
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
