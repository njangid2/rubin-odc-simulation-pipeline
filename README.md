# ODC Satellite Streak & Brightness Pipeline

Simulates the impact of SpaceX Orbital Data Center (ODC) satellite constellations on Rubin Observatory (LSST) observations. The pipeline propagates synthetic satellite populations through Rubin pointings, identifies field-of-view crossings, and estimates apparent brightness for each crossing using a BRDF-based reflectance model.

---

## Dependencies

```bash
pip install numpy pandas astropy sgp4 lumos
```

You will also need:
- `data_center_20deg.py` — local module (included in this repo); must be in the same directory as `brightness_20deg.py`
- `starlink` — satellite surface model library (used internally by `data_center_20deg.py`)
- `lumos` — BRDF brightness framework ([Fankhauser et al. 2023](https://github.com/Fankhauser/lumos))

> **Note:** `data_center_20deg.py` is not pip-installable. Keep it in the working directory alongside the step scripts.

---

## Input Data

Before running Step 1, you need a Rubin scheduler database CSV. The pipeline expects a file containing at minimum:

| Column | Description |
|---|---|
| `observationId` | Unique pointing identifier |
| `fieldRA` | Right ascension of pointing center (deg) |
| `fieldDec` | Declination of pointing center (deg) |
| `observationStartMJD` | Exposure start time (MJD) |
| `visitExposureTime` | Exposure duration (seconds) |
| `filter` | Photometric filter |
| `night` | Observing night index |

Pointing data can be obtained from the [Rubin baseline scheduler database](https://rubin-scheduler.lsst.io).

---

## Pipeline Overview

```
Step 1  →  Step 2  →  Step 3a  →  Step 3b  →  Step 4
[Filter]   [Propagate] [Sat Az/El] [Sun Az/El] [Brightness]
```

Each step produces output files that feed directly into the next step. **Steps must be run in order.**

---

## Step-by-Step Guide

### Step 1 — Affected Exposure Identification
**Script:** `Step1.py`  
**Output directory:** `streak_output_sgp4/`

Identifies which Rubin pointings are plausibly affected by each constellation shell. For each shell and each calendar day, one representative satellite per orbital plane is propagated across the full day using SGP4. A pointing is flagged if any representative satellite passes within 5° of the pointing center at any sampled time.

This is a fast prefilter — it favours recall over precision. Some flagged pointings will have no true crossing, but genuine crossings will not be missed.

Also generates and caches per-shell, per-day TLE `.pkl` files used by Step 2.

**Outputs:**
- `streak_output_sgp4/streak_results_shell_XXX.csv` — one file per shell, all pointings with `n_streaks > 0` flagged
- `streak_output_sgp4/tles_shell_XXX_YYYY-MM-DD.pkl` — synthetic TLE cache

```bash
python Step1.py
```

---

### Step 2 — Propagation and Field-of-View Matching
**Script:** `step2.py`  
**Input:** `streak_output_sgp4/`  
**Output directory:** `step2_output/`

For every pointing flagged in Step 1, propagates all satellites in that shell across the full exposure window. Each exposure is sampled at 10 uniformly-spaced time steps (~3 s resolution for a 30 s exposure). Satellite positions are transformed from TEME to RA/Dec and matched against the Rubin field-of-view radius of 1.75°.

Also computes a sunlit/shadow flag for each satellite at each time step using a cylindrical Earth shadow model.

**Outputs:**
- `step2_output/streak_trajectories_YYYY-MM-DD.csv` — one file per day, all FOV-matched rows with columns:
  `pointing_id`, `pointing_ra`, `pointing_dec`, `pointing_mjd`, `pointing_exptime`, `pointing_filter`, `pointing_night`, `shell_id`, `sat_name`, `step`, `t_mjd`, `ra_deg`, `dec_deg`, `sep_fov_center_deg`, `in_fov`, `sunlit`

```bash
python step2.py
```

---

### Step 3a — Satellite Azimuth/Elevation
**Script:** `step3a.py`  
**Input:** `step2_output/`  
**Output directory:** `step3_output/`

Transforms satellite RA/Dec from Step 2 into observer-frame azimuth and elevation at the Rubin site (Cerro Pachón, −30.2446°, −70.7494°, 2663 m).

**Outputs:**
- `step3_output/streak_trajectories_YYYY-MM-DD_azel.csv` — row-aligned sidecar file with columns: `az_deg`, `el_deg`

```bash
python step3a.py
```

---

### Step 3b — Sun Position
**Script:** `step3b.py`  
**Input:** `step2_output/`  
**Output directory:** `step3_output/`

Computes Sun azimuth and elevation at the Rubin site for every unique timestamp across all Step 2 trajectory files. Computed once per unique `t_mjd` value and stored as a lookup table, avoiding redundant recomputation across shells and rows.

**Outputs:**
- `step3_output/sun_positions.csv` — columns: `t_mjd`, `sun_az_deg`, `sun_el_deg`

```bash
python step3b.py
```

> Step 3a and Step 3b are independent and can be run in either order, but both must complete before Step 4.

---

### Step 4 — Brightness Modeling
**Script:** `brightness_20deg.py`  
**Requires:** `data_center_20deg.py` in the same directory  
**Input:** `step2_output/`, `step3_output/`  
**Output directory:** `step4_output_20deg/`

Computes apparent AB magnitude for every sunlit, in-field satellite crossing. Brightness is derived from the instantaneous Sun–satellite–observer geometry using a BRDF-based reflectance model (Lumos-Sat framework). The satellite is modeled with:
- **Chassis:** Starlink V2 Mini BRDF scaled to ~7 × 3.5 m²
- **Solar array:** Starlink V1.5 BRDF scaled to 400 kW power (1,679 m² panel area)
- **Panel offset:** 20° from Sun toward nadir (Rodrigues rotation)

Brightness is set to `NaN` for rows that are shadowed, outside the FOV, or missing geometry data.

**Outputs:**
- `step4_output_20deg/streak_trajectories_YYYY-MM-DD_bright.csv` — row-aligned sidecar file with column: `ab_magnitude`

```bash
python brightness_20deg.py
```

---

## Joining All Outputs

To assemble the full per-day table:

```python
import pandas as pd

date = "2025-11-01"
df     = pd.read_csv(f"step2_output/streak_trajectories_{date}.csv")
azel   = pd.read_csv(f"step3_output/streak_trajectories_{date}_azel.csv")
sun    = pd.read_csv("step3_output/sun_positions.csv").set_index("t_mjd")
bright = pd.read_csv(f"step4_output_20deg/streak_trajectories_{date}_bright.csv")

df["az_deg"]       = azel["az_deg"]
df["el_deg"]       = azel["el_deg"]
df["sun_az_deg"]   = df["t_mjd"].map(sun["sun_az_deg"])
df["sun_el_deg"]   = df["t_mjd"].map(sun["sun_el_deg"])
df["ab_magnitude"] = bright["ab_magnitude"]
```

---

## Resumability

All steps are resume-safe. If a run is interrupted:
- **Step 1:** skips shells whose output CSV already exists
- **Step 2:** skips entire days whose merged CSV exists; skips individual shells whose intermediate CSV exists
- **Step 3a/3b:** skips dates/files already computed
- **Step 4:** skips days where the output row count matches the input row count

Simply re-run the same script to continue from where it stopped.

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

- All steps support parallel processing. Worker counts are set near the bottom of each script's `main()` call and can be adjusted to match available cores.
- TLE `.pkl` files from Step 1 can be deleted after Step 2 completes by setting `delete_tles=True` in `step2.py`.
- The brightness model clips AB magnitudes at approximately 12 mag for unphysical low-intensity cases.
- Earthshine is not included in the brightness model; only direct solar illumination is considered.
