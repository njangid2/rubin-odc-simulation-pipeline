"""
step3b_sun_positions.py
─────────────────────────────────────────────────────────────────────────────
Computes Sun azimuth and elevation at Rubin Observatory for every unique
t_mjd value that appears across all step2 trajectory CSVs.

OUTPUT
──────
    step3_output/sun_positions.csv

Columns:
    t_mjd        — UTC MJD (matches t_mjd in step2 files exactly)
    sun_az_deg   — Sun azimuth at Rubin (0=N 90=E 180=S 270=W)
    sun_el_deg   — Sun elevation at Rubin (negative = below horizon = night)

Usage later (join onto any step2 day CSV):
    df  = pd.read_csv("streak_trajectories_2025-11-01.csv")
    sun = pd.read_csv("step3_output/sun_positions.csv").set_index("t_mjd")
    df["sun_az_deg"] = df["t_mjd"].map(sun["sun_az_deg"])
    df["sun_el_deg"] = df["t_mjd"].map(sun["sun_el_deg"])

WHY ONE FILE FOR ALL DAYS
─────────────────────────
The Sun position only depends on time, not on which satellite or pointing.
Collecting all unique t_mjd values first and computing once is far more
efficient than recomputing per day or per row.

Typical numbers:
    ~365 days × ~700 pointings/day × 10 steps = ~2.5M rows in step2
    But many t_mjd values repeat across shells for the same pointing
    Unique t_mjd values are typically ~10-50% of total rows
    → Sun position computed once per unique time, not once per row

RESUMABILITY
────────────
If sun_positions.csv already exists → skip entirely.
Use force=True in main() to recompute.

USAGE
─────
    python step3b_sun_positions.py

Requirements:
    pip install astropy
"""

import os, gc, glob, time, logging, sys
import numpy as np
import pandas as pd
import astropy.time
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str, log_filename: str = "step3b.log") -> str:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt     = "%(asctime)s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return log_path

log = logging.info

# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

def _hms(s):
    h, r = divmod(int(s), 3600); m, sec = divmod(r, 60); frac = s - int(s)
    if h:  return f"{h}h {m}m {sec+frac:.2f}s"
    if m:  return f"{m}m {sec+frac:.2f}s"
    return f"{sec+frac:.2f}s"

# ─────────────────────────────────────────────────────────────────────────────
# Rubin Observatory
# ─────────────────────────────────────────────────────────────────────────────

RUBIN_LAT_DEG = -30.244639
RUBIN_LON_DEG = -70.749417
RUBIN_ELEV_M  =  2663.0

RUBIN = EarthLocation(
    lat    = RUBIN_LAT_DEG * u.deg,
    lon    = RUBIN_LON_DEG * u.deg,
    height = RUBIN_ELEV_M  * u.m,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sun Az/El computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_sun_azel(mjd_utc:   np.ndarray,
                     location:  EarthLocation,
                     batch_size: int = 10_000) -> tuple:
    """
    Compute Sun azimuth and elevation at location for each MJD timestamp.

    Parameters
    ----------
    mjd_utc    : 1-D array of unique MJD values (UTC)
    location   : astropy EarthLocation
    batch_size : timestamps per astropy batch

    Returns
    -------
    sun_az_deg : np.ndarray float32  (N,)
    sun_el_deg : np.ndarray float32  (N,)
                 Negative values = Sun below horizon = night
    """
    N          = len(mjd_utc)
    sun_az_out = np.empty(N, dtype=np.float32)
    sun_el_out = np.empty(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end      = min(start + batch_size, N)
        sl       = slice(start, end)

        obs_time  = astropy.time.Time(
            mjd_utc[sl], format="mjd", scale="utc"
        )
        frame     = AltAz(obstime=obs_time, location=location)

        # get_sun returns Sun position in GCRS
        # transform_to(AltAz) gives topocentric Az/El at observer location
        sun_coord = get_sun(obs_time)
        sun_altaz = sun_coord.transform_to(frame)

        sun_az_out[sl] = sun_altaz.az.deg.astype(np.float32)
        sun_el_out[sl] = sun_altaz.alt.deg.astype(np.float32)

        log(f"  [Sun] {end:,}/{N:,} timestamps computed …")

    return sun_az_out, sun_el_out

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(
    step2_dirs   = (
        "step2_output",
    ),
    output_dir   = "step3_output",
    output_file  = "sun_positions.csv",
    batch_size   = 10_000,
    log_filename = "step3b.log",
    force        = False,       # set True to recompute even if file exists
):
    log_path   = setup_logging(output_dir, log_filename)
    wall_start = time.perf_counter()

    log("=" * 70)
    log("  STEP 3b: Sun Positions  (unique t_mjd → sun_az_deg, sun_el_deg)")
    log("=" * 70)
    log(f"  output_dir  : {output_dir}")
    log(f"  output_file : {output_file}")
    log(f"  batch_size  : {batch_size:,}")
    log(f"  Rubin       : lat={RUBIN_LAT_DEG}°  "
        f"lon={RUBIN_LON_DEG}°  elev={RUBIN_ELEV_M}m")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_file)

    # ── Resume check ──────────────────────────────────────────────────────
    if os.path.isfile(out_path) and not force:
        existing = pd.read_csv(out_path)
        log(f"[SKIP] {out_path} already exists "
            f"({len(existing):,} rows) — set force=True to recompute.")
        return
    elif os.path.isfile(out_path) and force:
        log(f"[FORCE] Recomputing — overwriting existing {out_path}")

    # ── Collect all step2 day CSVs ─────────────────────────────────────────
    all_day_csvs = []
    for d in step2_dirs:
        found = sorted(glob.glob(
            os.path.join(d, "streak_trajectories_*.csv")
        ))
        log(f"  {d}: {len(found)} CSV(s)")
        all_day_csvs.extend(found)

    all_day_csvs = sorted(all_day_csvs, key=lambda p: os.path.basename(p))

    if not all_day_csvs:
        log("[WARN] No step2 CSVs found — check step2_dirs.")
        return

    log(f"\n[INFO] {len(all_day_csvs)} day CSV(s) found")

    # ── Collect all unique t_mjd values — load only t_mjd column ──────────
    log("\n[STEP 1] Collecting unique t_mjd values across all days …")
    t0 = time.perf_counter()

    all_mjds = set()
    total_rows = 0

    for i, path in enumerate(all_day_csvs):
        date_str = (os.path.basename(path)
                    .replace("streak_trajectories_", "")
                    .replace(".csv", ""))
        try:
            # Load only t_mjd — minimal RAM
            mjds = pd.read_csv(path, usecols=["t_mjd"])["t_mjd"].values
            all_mjds.update(mjds.tolist())
            total_rows += len(mjds)

            if (i + 1) % 30 == 0 or i == len(all_day_csvs) - 1:
                log(f"  {i+1}/{len(all_day_csvs)} days scanned  "
                    f"unique t_mjd so far: {len(all_mjds):,}")
            del mjds
            gc.collect()

        except Exception as e:
            log(f"  [WARN] Could not read {date_str}: {e}")

    log(f"\n  Total rows across all days : {total_rows:,}")
    log(f"  Unique t_mjd values        : {len(all_mjds):,}  "
        f"({100*len(all_mjds)/max(total_rows,1):.1f}% of total rows)")
    log(f"  Scan time                  : {_hms(time.perf_counter()-t0)}")

    # Sort unique MJDs for clean output
    unique_mjds = np.array(sorted(all_mjds), dtype=np.float64)
    del all_mjds
    gc.collect()

    log(f"\n  MJD range: {unique_mjds[0]:.6f} → {unique_mjds[-1]:.6f}")

    # ── Compute Sun Az/El for all unique timestamps ────────────────────────
    log(f"\n[STEP 2] Computing Sun Az/El for {len(unique_mjds):,} "
        f"unique timestamps …")
    t0 = time.perf_counter()

    sun_az, sun_el = compute_sun_azel(
        mjd_utc    = unique_mjds,
        location   = RUBIN,
        batch_size = batch_size,
    )

    log(f"  Sun computation time: {_hms(time.perf_counter()-t0)}")

    # Quick sanity check
    n_day_obs   = int((sun_el > 0).sum())
    n_twilight  = int(((sun_el > -18) & (sun_el <= 0)).sum())
    n_night     = int((sun_el <= -18).sum())
    log(f"\n  Sun elevation breakdown:")
    log(f"    Day (el > 0°)         : {n_day_obs:,}  "
        f"({100*n_day_obs/len(unique_mjds):.1f}%)")
    log(f"    Twilight (-18° to 0°) : {n_twilight:,}  "
        f"({100*n_twilight/len(unique_mjds):.1f}%)")
    log(f"    Night (el < -18°)     : {n_night:,}  "
        f"({100*n_night/len(unique_mjds):.1f}%)")

    # ── Save ──────────────────────────────────────────────────────────────
    log(f"\n[STEP 3] Saving {out_path} …")

    sun_df = pd.DataFrame({
        "t_mjd":       np.round(unique_mjds, 9),
        "sun_az_deg":  np.round(sun_az, 4).astype(np.float32),
        "sun_el_deg":  np.round(sun_el, 4).astype(np.float32),
    })
    sun_df.to_csv(out_path, index=False)

    size_mb = os.path.getsize(out_path) / 1e6
    log(f"[SAVED] {out_path}  "
        f"({len(sun_df):,} rows  {size_mb:.2f} MB)")

    del sun_df, unique_mjds, sun_az, sun_el
    gc.collect()

    # ── Summary ────────────────────────────────────────────────────────────
    log("")
    log("=" * 70)
    log(f"  Unique timestamps : {len(unique_mjds) if 'unique_mjds' in dir() else 'freed'}")
    log(f"  Output size       : {size_mb:.2f} MB")
    log(f"  Total time        : {_hms(time.perf_counter() - wall_start)}")
    log("")
    log("  To join onto a step2 day CSV:")
    log("      df  = pd.read_csv('streak_trajectories_YYYY-MM-DD.csv')")
    log(f"      sun = pd.read_csv('{out_path}').set_index('t_mjd')")
    log("      df['sun_az_deg'] = df['t_mjd'].map(sun['sun_az_deg'])")
    log("      df['sun_el_deg'] = df['t_mjd'].map(sun['sun_el_deg'])")
    log("")
    log("[DONE]")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(
        step2_dirs  = (
            "step2_output",
        ),
        output_dir  = "step3_output",
        output_file = "sun_positions.csv",
        batch_size  = 10_000,
        log_filename= "step3b.log",
        force       = False,    # True → recompute even if file exists
    )