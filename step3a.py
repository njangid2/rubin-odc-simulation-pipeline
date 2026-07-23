"""
step3_add_azel.py  (v3 — one-to-one aligned, sunlit-only compute, parallel)
─────────────────────────────────────────────────────────────────────────────
OUTPUT: strictly row-aligned with step2 CSV.
  - Same number of rows as step2 file
  - Row N in azel = Row N in step2 (no join needed)
  - az_deg=NaN, el_deg=NaN for non-sunlit or out-of-fov rows
  - Az/El only COMPUTED for sunlit+in_fov rows (fast), rest filled with NaN

To use:
    df         = pd.read_csv("streak_trajectories_2025-11-01.csv")
    azel       = pd.read_csv("streak_trajectories_2025-11-01_azel.csv")
    df["az_deg"] = azel["az_deg"]   # direct assign — no merge needed
    df["el_deg"] = azel["el_deg"]

    # Then filter to sunlit only for analysis:
    sunlit_df = df[df["sunlit"] == True]   # az/el will be valid here

PARALLELISM
───────────
Days processed in parallel with Pool.map (n_workers days at once).

RESUMABILITY
────────────
If azel output already exists for a day → skip it.

IERS WARNING
────────────
Suppressed — arcsecond precision not needed for streak work.
"""

import os, gc, glob, time, logging, sys, warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
import astropy.time
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from astropy.utils import iers

# ─────────────────────────────────────────────────────────────────────────────
# Suppress IERS warning
# ─────────────────────────────────────────────────────────────────────────────
iers.conf.auto_download = False
iers.conf.auto_max_age  = None
warnings.filterwarnings("ignore", message=".*polar motions.*")
warnings.filterwarnings("ignore", message=".*IERS.*")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str, log_filename: str = "step3.log") -> str:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt     = "%(asctime)s  [PID %(process)5d]  %(message)s",
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
# Az/El conversion (batched)
# ─────────────────────────────────────────────────────────────────────────────

def compute_azel(ra_deg:  np.ndarray,
                 dec_deg: np.ndarray,
                 mjd_utc: np.ndarray,
                 location: EarthLocation,
                 batch_size: int = 50_000) -> tuple:
    """
    Convert ICRS RA/Dec + UTC MJD → topocentric Az/El.
    Only called on the sunlit+in_fov subset — fast.
    """
    N      = len(ra_deg)
    az_out = np.empty(N, dtype=np.float32)
    el_out = np.empty(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sl  = slice(start, end)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obs_time = astropy.time.Time(
                mjd_utc[sl], format="mjd", scale="utc"
            )
            frame = AltAz(obstime=obs_time, location=location)
            sky   = SkyCoord(
                ra    = ra_deg[sl]  * u.deg,
                dec   = dec_deg[sl] * u.deg,
                frame = ICRS(),
            )
            altaz = sky.transform_to(frame)

        az_out[sl] = altaz.az.deg.astype(np.float32)
        el_out[sl] = altaz.alt.deg.astype(np.float32)

    return az_out, el_out

# ─────────────────────────────────────────────────────────────────────────────
# Worker: process ONE day
# ─────────────────────────────────────────────────────────────────────────────

def process_day_worker(args):
    """
    Worker function — one day CSV → one azel CSV, row-aligned with step2.
    Non-sunlit rows get NaN az/el. Sunlit+in_fov rows get computed values.
    args = (input_path, output_path, batch_size, log_path, output_dir)
    """
    input_path, output_path, batch_size, log_path, output_dir = args

    # Re-init logging + suppress warnings in worker
    setup_logging(output_dir, os.path.basename(log_path))
    iers.conf.auto_download = False
    iers.conf.auto_max_age  = None
    warnings.filterwarnings("ignore", message=".*polar motions.*")
    warnings.filterwarnings("ignore", message=".*IERS.*")

    date_str = (os.path.basename(input_path)
                .replace("streak_trajectories_", "")
                .replace(".csv", ""))

    # Resume check
    if os.path.isfile(output_path):
        size_mb = os.path.getsize(output_path) / 1e6
        log(f"[SKIP] {date_str} — already exists ({size_mb:.2f} MB)")
        return {"date": date_str, "status": "skip", "n_rows": 0}

    t0 = time.perf_counter()
    log(f"[START] {date_str}")

    try:
        # Load only needed columns
        df = pd.read_csv(
            input_path,
            usecols=["ra_deg", "dec_deg", "t_mjd", "in_fov", "sunlit"]
        )
        N = len(df)
        log(f"  {date_str} | {N:,} total rows")

        # ── Allocate output arrays — NaN by default ───────────────────────
        # Every row gets a slot; non-sunlit rows stay NaN
        az_full = np.full(N, np.nan, dtype=np.float32)
        el_full = np.full(N, np.nan, dtype=np.float32)

        if N == 0:
            pd.DataFrame({
                "az_deg": az_full,
                "el_deg": el_full,
            }).to_csv(output_path, index=False)
            log(f"  {date_str} | empty file written")
            return {"date": date_str, "status": "done",
                    "n_rows": 0, "n_sunlit": 0}

        # ── Find sunlit + in_fov rows ─────────────────────────────────────
        sunlit_mask = (
            (df["in_fov"].astype(str).str.lower() == "true") &
            (df["sunlit"].astype(str).str.lower() == "true")
        )
        sunlit_idx = np.where(sunlit_mask.values)[0]
        n_sunlit   = len(sunlit_idx)

        log(f"  {date_str} | {n_sunlit:,}/{N:,} rows are sunlit+in_fov "
            f"({100*n_sunlit/N:.1f}%) — computing Az/El for these only")

        if n_sunlit > 0:
            df_sunlit = df.iloc[sunlit_idx]

            az_sub, el_sub = compute_azel(
                ra_deg    = df_sunlit["ra_deg"].values.astype(np.float64),
                dec_deg   = df_sunlit["dec_deg"].values.astype(np.float64),
                mjd_utc   = df_sunlit["t_mjd"].values.astype(np.float64),
                location  = RUBIN,
                batch_size= batch_size,
            )

            # Place computed values back at their original row positions
            az_full[sunlit_idx] = az_sub
            el_full[sunlit_idx] = el_sub

            del df_sunlit, az_sub, el_sub

            # Sanity check
            n_neg_el = int((el_full[sunlit_idx] < 0).sum())
            n_nan_az = int(np.isnan(az_full[sunlit_idx]).sum())
            if n_neg_el > 0:
                log(f"  [WARN] {date_str} | {n_neg_el:,} sunlit rows "
                    f"have el_deg < 0")
            if n_nan_az > 0:
                log(f"  [WARN] {date_str} | {n_nan_az:,} NaN az_deg values")

        del df
        gc.collect()

        # ── Write output — same number of rows as step2 ───────────────────
        azel_df = pd.DataFrame({
            "az_deg": np.round(az_full, 4),   # NaN for non-sunlit rows
            "el_deg": np.round(el_full, 4),   # NaN for non-sunlit rows
        })
        azel_df.to_csv(output_path, index=False)
        del azel_df, az_full, el_full
        gc.collect()

        size_mb = os.path.getsize(output_path) / 1e6
        elapsed = time.perf_counter() - t0
        log(f"  [DONE] {date_str} | {N:,} rows written  "
            f"({n_sunlit:,} with Az/El, rest NaN)  "
            f"{size_mb:.2f} MB  {_hms(elapsed)}")

        return {
            "date":     date_str,
            "status":   "done",
            "n_rows":   N,
            "n_sunlit": n_sunlit,
            "size_mb":  size_mb,
            "elapsed":  elapsed,
        }

    except Exception as e:
        import traceback
        log(f"  [ERROR] {date_str}: {e}\n{traceback.format_exc()}")
        return {"date": date_str, "status": "error", "n_rows": 0}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(
    step2_dirs   = (
        "step2_output",
    ),
    output_dir   = "step3_output",
    batch_size   = 50_000,
    n_workers    = 8,
    log_filename = "step3.log",
):
    log_path   = setup_logging(output_dir, log_filename)
    wall_start = time.perf_counter()

    log("=" * 70)
    log("  STEP 3: Compute Az/El  (row-aligned, sunlit-only compute, parallel)")
    log("=" * 70)
    log(f"  output_dir  : {output_dir}")
    log(f"  batch_size  : {batch_size:,}")
    log(f"  n_workers   : {n_workers}")
    log(f"  alignment   : one-to-one with step2 (NaN for non-sunlit rows)")
    log(f"  IERS warn   : suppressed")
    log(f"  Rubin       : lat={RUBIN_LAT_DEG}°  "
        f"lon={RUBIN_LON_DEG}°  elev={RUBIN_ELEV_M}m")
    log(f"  log file    : {log_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Collect all input day CSVs ─────────────────────────────────────────
    all_input_csvs = []
    for d in step2_dirs:
        found = sorted(glob.glob(
            os.path.join(d, "streak_trajectories_*.csv")
        ))
        log(f"  {d}: {len(found)} CSV(s)")
        all_input_csvs.extend(found)

    all_input_csvs = sorted(
        all_input_csvs, key=lambda p: os.path.basename(p)
    )

    if not all_input_csvs:
        log("[WARN] No input CSVs found — check step2_dirs paths.")
        return

    log(f"\n[INFO] {len(all_input_csvs)} day CSV(s) to process "
        f"using {n_workers} workers\n")

    # ── Build worker args ──────────────────────────────────────────────────
    worker_args = []
    for input_path in all_input_csvs:
        date_str = (os.path.basename(input_path)
                    .replace("streak_trajectories_", "")
                    .replace(".csv", ""))
        output_path = os.path.join(
            output_dir,
            f"streak_trajectories_{date_str}_azel.csv"
        )
        worker_args.append(
            (input_path, output_path, batch_size, log_path, output_dir)
        )

    # ── Parallel dispatch ──────────────────────────────────────────────────
    with Pool(processes=n_workers) as pool:
        results = pool.map(process_day_worker, worker_args)

    # ── Summary ────────────────────────────────────────────────────────────
    n_done  = sum(1 for r in results if r["status"] == "done")
    n_skip  = sum(1 for r in results if r["status"] == "skip")
    n_error = sum(1 for r in results if r["status"] == "error")
    total_rows   = sum(r.get("n_rows",   0) for r in results)
    total_sunlit = sum(r.get("n_sunlit", 0) for r in results)

    done_csvs = sorted(glob.glob(
        os.path.join(output_dir, "streak_trajectories_*_azel.csv")
    ))
    total_gb = sum(os.path.getsize(p) for p in done_csvs) / 1e9

    log("")
    log("=" * 70)
    log(f"  Days processed    : {n_done}")
    log(f"  Days skipped      : {n_skip}")
    log(f"  Days errored      : {n_error}")
    log(f"  Total rows        : {total_rows:,}  (matches step2)")
    log(f"  Sunlit rows       : {total_sunlit:,}  (have valid Az/El)")
    log(f"  Non-sunlit rows   : {total_rows-total_sunlit:,}  (NaN az/el)")
    log(f"  Output files      : {len(done_csvs)}  ({total_gb:.3f} GB total)")
    log(f"  Total time        : {_hms(time.perf_counter() - wall_start)}")
    log("")
    log("[DONE]")
    log(f"[LOG]  {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(
        step2_dirs  = (
            "step2_output",
        ),
        output_dir  = "step3_output",
        batch_size  = 50_000,
        n_workers   = 94,        # match --cpus-per-task in sbatch
        log_filename= "step3a.log",
    )