"""
step4_brightness.py  (v2 — corrected paths + all 94 shells)
─────────────────────────────────────────────────────────────────────────────
Computes satellite brightness for every row in each step2 day CSV and saves
a single-column sidecar file — positionally aligned with the step2 CSV.

    streak_trajectories_YYYY-MM-DD.csv        ← step2  (untouched)
    streak_trajectories_YYYY-MM-DD_azel.csv   ← step3  (az_deg, el_deg)
    streak_trajectories_YYYY-MM-DD_bright.csv ← step4  (ab_magnitude only)

To join all sidecar files for one day:
    df     = pd.read_csv("streak_trajectories_2025-11-01.csv")
    azel   = pd.read_csv("streak_trajectories_2025-11-01_azel.csv")
    bright = pd.read_csv("streak_trajectories_2025-11-01_bright.csv")
    df["az_deg"]       = azel["az_deg"]
    df["el_deg"]       = azel["el_deg"]
    df["ab_magnitude"] = bright["ab_magnitude"]

USAGE
─────
    python step4_brightness.py

Requirements:
    pip install data_center
"""

import os, gc, glob, time, logging, sys, traceback
from multiprocessing import Pool
import numpy as np
import pandas as pd
import data_center_20deg as data_center

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str, log_filename: str = "step4_20deg.log") -> str:
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
# Shell id → orbital altitude (km)  — ALL 94 SXODC shells
# ─────────────────────────────────────────────────────────────────────────────

SHELL_ALT_KM = {
    # LEO-30: shells 1–25  (~686–718 km, ~30° inc)
     1:  686.0,   2:  687.3,   3:  688.7,   4:  690.0,   5:  691.3,
     6:  692.7,   7:  694.0,   8:  695.3,   9:  696.7,  10:  698.0,
    11:  699.3,  12:  700.7,  13:  702.0,  14:  703.3,  15:  704.7,
    16:  706.0,  17:  707.3,  18:  708.7,  19:  710.0,  20:  711.3,
    21:  712.7,  22:  714.0,  23:  715.3,  24:  716.7,  25:  718.0,
    # MEO-30: shells 26–50  (~946–978 km, ~30° inc)
    26:  946.0,  27:  947.3,  28:  948.7,  29:  950.0,  30:  951.3,
    31:  952.7,  32:  954.0,  33:  955.3,  34:  956.7,  35:  958.0,
    36:  959.3,  37:  960.7,  38:  962.0,  39:  963.3,  40:  964.7,
    41:  966.0,  42:  967.3,  43:  968.7,  44:  970.0,  45:  971.3,
    46:  972.7,  47:  974.0,  48:  975.3,  49:  976.7,  50:  978.0,
    # SSO-97: shells 51–72  (~707–744 km, ~97–98° inc)
    51:  707.0,  52:  708.8,  53:  710.5,  54:  712.3,  55:  714.0,
    56:  715.8,  57:  717.6,  58:  719.3,  59:  721.1,  60:  722.9,
    61:  724.6,  62:  726.4,  63:  728.1,  64:  729.9,  65:  731.7,
    66:  733.4,  67:  735.2,  68:  737.0,  69:  738.7,  70:  740.5,
    71:  742.2,  72:  744.0,
    # SSO-99: shells 73–94  (~967–1002 km, ~99.4–99.5° inc)
    73:  967.0,  74:  968.7,  75:  970.3,  76:  972.0,  77:  973.7,
    78:  975.3,  79:  977.0,  80:  978.7,  81:  980.3,  82:  982.0,
    83:  983.7,  84:  985.3,  85:  987.0,  86:  988.7,  87:  990.3,
    88:  992.0,  89:  993.7,  90:  995.3,  91:  997.0,  92:  998.7,
    93: 1000.3,  94: 1002.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Brightness for one row
# ─────────────────────────────────────────────────────────────────────────────

def calc_brightness_row(
    sat_alt_km:  float,
    sat_el_deg:  float,
    sat_az_deg:  float,
    sun_el_deg:  float,
    sun_az_deg:  float,
    power_kw:    float,
    continuous:  bool,
) -> float:
    """
    Call data_center.calculate_brightness for one row.
    Returns ab_magnitude as float, or NaN on failure.
    """
    try:
        result = data_center.calculate_brightness(
            sat_height   = sat_alt_km * 1e3,   # km → metres
            sat_altitude = sat_el_deg,
            sat_azimuth  = sat_az_deg,
            sun_altitude = sun_el_deg,
            sun_azimuth  = sun_az_deg,
            power_kw     = power_kw,
            continuous   = continuous,
            offset_deg = 20,
        )
        return float(np.atleast_1d(result["ab_magnitude"])[0])
    except Exception:
        return float("nan")

# ─────────────────────────────────────────────────────────────────────────────
# Worker function — processes ONE day
# ─────────────────────────────────────────────────────────────────────────────

def process_day(
    step2_path:  str,
    azel_path:   str,
    output_path: str,
    sun_dict:    dict,
    power_kw:    float,
    continuous:  bool,
    output_dir:  str,
    log_path:    str,
):
    # Re-init logging in worker
    setup_logging(output_dir, os.path.basename(log_path))

    date_str = (os.path.basename(step2_path)
                .replace("streak_trajectories_", "")
                .replace(".csv", ""))

    t0 = time.perf_counter()
    log(f"[START] {date_str}")

    # ── Resume check ───────────────────────────────────────────────────────
    if os.path.isfile(output_path):
        try:
            n_step2  = sum(1 for _ in open(step2_path))  - 1
            n_bright = sum(1 for _ in open(output_path)) - 1
            if n_step2 == n_bright:
                size_mb = os.path.getsize(output_path) / 1e6
                log(f"[SKIP]  {date_str} — complete ({n_bright:,} rows  "
                    f"{size_mb:.2f} MB)")
                return {"date": date_str, "status": "skipped",
                        "n_rows": n_bright, "elapsed": 0.0}
            else:
                log(f"[REDO]  {date_str} — row mismatch "
                    f"(step2={n_step2} bright={n_bright}) — reprocessing")
        except Exception:
            log(f"[REDO]  {date_str} — could not verify — reprocessing")

    # ── Check azel file exists ─────────────────────────────────────────────
    if not os.path.isfile(azel_path):
        log(f"[WARN]  {date_str} — azel file missing: {azel_path}")
        return {"date": date_str, "status": "skipped_no_azel",
                "n_rows": 0, "elapsed": 0.0}

    # ── Load step2 ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(
            step2_path,
            usecols=["shell_id", "t_mjd", "sunlit"]
        )
    except Exception as e:
        log(f"[ERROR] {date_str} — step2 read failed: {e}")
        return {"date": date_str, "status": "error", "n_rows": 0, "elapsed": 0.0}

    N = len(df)
    log(f"  {date_str} | {N:,} rows")

    if N == 0:
        pd.DataFrame({"ab_magnitude": pd.Series(dtype=np.float32)}
                     ).to_csv(output_path, index=False)
        return {"date": date_str, "status": "done_empty",
                "n_rows": 0, "elapsed": 0.0}

    # ── Load azel (row-aligned) ────────────────────────────────────────────
    try:
        azel = pd.read_csv(azel_path, usecols=["az_deg", "el_deg"])
    except Exception as e:
        log(f"[ERROR] {date_str} — azel read failed: {e}")
        return {"date": date_str, "status": "error", "n_rows": 0, "elapsed": 0.0}

    if len(azel) != N:
        log(f"[ERROR] {date_str} — row mismatch step2={N} azel={len(azel)}")
        return {"date": date_str, "status": "error", "n_rows": 0, "elapsed": 0.0}

    df["az_deg"] = azel["az_deg"].values
    df["el_deg"] = azel["el_deg"].values
    del azel
    gc.collect()

    # ── Join sun positions ─────────────────────────────────────────────────
    mjd_vals   = df["t_mjd"].values
    sun_az_arr = np.array(
        [sun_dict.get(float(m), (np.nan, np.nan))[0] for m in mjd_vals],
        dtype=np.float32
    )
    sun_el_arr = np.array(
        [sun_dict.get(float(m), (np.nan, np.nan))[1] for m in mjd_vals],
        dtype=np.float32
    )

    n_missing_sun = int(np.isnan(sun_az_arr).sum())
    if n_missing_sun > 0:
        log(f"  {date_str} | [WARN] {n_missing_sun:,} rows missing sun position")

    # ── Map shell_id → altitude ───────────────────────────────────────────
    df["alt_km"] = df["shell_id"].map(SHELL_ALT_KM)
    n_unknown_shell = int(df["alt_km"].isna().sum())
    if n_unknown_shell > 0:
        bad_shells = df[df["alt_km"].isna()]["shell_id"].unique().tolist()
        log(f"  {date_str} | [WARN] {n_unknown_shell:,} rows have unknown "
            f"shell_id: {bad_shells}")

    # ── Build validity mask ───────────────────────────────────────────────
    sunlit_mask = df["sunlit"].astype(str).str.lower() == "true"
    valid_mask  = (
        sunlit_mask &
        df["az_deg"].notna() &
        df["el_deg"].notna() &
        df["alt_km"].notna() &
        ~np.isnan(sun_az_arr) &
        ~np.isnan(sun_el_arr)
    )

    n_sunlit = int(sunlit_mask.sum())
    n_valid  = int(valid_mask.sum())
    log(f"  {date_str} | sunlit={n_sunlit:,}  valid for brightness={n_valid:,}")

    # ── Compute brightness — NaN for non-valid rows ───────────────────────
    ab_mag    = np.full(N, np.nan, dtype=np.float32)
    valid_idx = np.where(valid_mask.values)[0]
    alt_arr   = df["alt_km"].values
    el_arr    = df["el_deg"].values
    az_arr    = df["az_deg"].values

    for i, row_idx in enumerate(valid_idx):
        ab_mag[row_idx] = calc_brightness_row(
            sat_alt_km = float(alt_arr   [row_idx]),
            sat_el_deg = float(el_arr    [row_idx]),
            sat_az_deg = float(az_arr    [row_idx]),
            sun_el_deg = float(sun_el_arr[row_idx]),
            sun_az_deg = float(sun_az_arr[row_idx]),
            power_kw   = power_kw,
            continuous = continuous,
        )
        if (i + 1) % 10_000 == 0 or i == len(valid_idx) - 1:
            log(f"  {date_str} | brightness {i+1:,}/{n_valid:,} …")

    del df, alt_arr, el_arr, az_arr, sun_az_arr, sun_el_arr
    gc.collect()

    # ── Save single-column output (row-aligned with step2) ────────────────
    pd.DataFrame({
        "ab_magnitude": np.round(ab_mag, 6).astype(np.float32)
    }).to_csv(output_path, index=False)

    size_mb = os.path.getsize(output_path) / 1e6
    elapsed = time.perf_counter() - t0
    n_nan   = int(np.isnan(ab_mag).sum())

    log(f"[DONE]  {date_str} | {N:,} rows  computed={n_valid:,}  "
        f"NaN={n_nan:,}  {size_mb:.2f} MB  {_hms(elapsed)}")

    del ab_mag
    gc.collect()

    return {
        "date":    date_str,
        "status":  "done",
        "n_rows":  N,
        "n_valid": n_valid,
        "n_nan":   n_nan,
        "size_mb": size_mb,
        "elapsed": elapsed,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(
    step2_dir    = "step2_output",          # ← single folder (same dir as script)
    step3_dir    = "step3_output",
    sun_csv      = "step3_output/sun_positions.csv",
    output_dir   = "step4_output_20deg",
    power_kw     = 400.0,
    continuous   = False,
    n_workers    = 8,
    log_filename = "step4_20deg.log",
):
    log_path   = setup_logging(output_dir, log_filename)
    wall_start = time.perf_counter()

    log("=" * 70)
    log("  STEP 4: Brightness  (parallel days, single-column sidecar)")
    log("=" * 70)
    log(f"  step2_dir   : {step2_dir}")
    log(f"  step3_dir   : {step3_dir}")
    log(f"  sun_csv     : {sun_csv}")
    log(f"  output_dir  : {output_dir}")
    log(f"  power_kw    : {power_kw}")
    log(f"  continuous  : {continuous}")
    log(f"  n_workers   : {n_workers}")
    log(f"  shells      : {len(SHELL_ALT_KM)} shells mapped "
        f"(ids {min(SHELL_ALT_KM)}–{max(SHELL_ALT_KM)})")
    log(f"  log file    : {log_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Load sun positions once ────────────────────────────────────────────
    log(f"\n[INFO] Loading sun positions from {sun_csv} …")
    if not os.path.isfile(sun_csv):
        raise FileNotFoundError(
            f"Sun positions not found: {sun_csv}\n"
            "Run step3b_sun_positions.py first."
        )
    sun_raw  = pd.read_csv(sun_csv)
    sun_dict = {
        float(row.t_mjd): (float(row.sun_az_deg), float(row.sun_el_deg))
        for row in sun_raw.itertuples()
    }
    del sun_raw
    gc.collect()
    log(f"  {len(sun_dict):,} sun timestamps loaded")

    # ── Collect step2 day CSVs ─────────────────────────────────────────────
    all_step2_csvs = sorted(glob.glob(
        os.path.join(step2_dir, "streak_trajectories_*.csv")
    ))
    # Exclude any azel files if they ended up in same folder
    all_step2_csvs = [p for p in all_step2_csvs if "_azel" not in p]

    if not all_step2_csvs:
        log(f"[WARN] No step2 CSVs found in {step2_dir}")
        return

    log(f"\n[INFO] {len(all_step2_csvs)} day CSV(s) found in {step2_dir}")

    # ── Build worker args ──────────────────────────────────────────────────
    worker_args = []
    for step2_path in all_step2_csvs:
        date_str    = (os.path.basename(step2_path)
                       .replace("streak_trajectories_", "")
                       .replace(".csv", ""))
        azel_path   = os.path.join(
            step3_dir,
            f"streak_trajectories_{date_str}_azel.csv"
        )
        output_path = os.path.join(
            output_dir,
            f"streak_trajectories_{date_str}_bright.csv"
        )
        worker_args.append((
            step2_path, azel_path, output_path,
            sun_dict, power_kw, continuous,
            output_dir, log_path,
        ))

    log(f"[INFO] Dispatching {len(worker_args)} days to {n_workers} workers …\n")

    # ── Parallel dispatch ──────────────────────────────────────────────────
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(process_day, worker_args)

    # ── Summary ────────────────────────────────────────────────────────────
    log("")
    log("=" * 70)
    log("  RESULTS SUMMARY")
    log("=" * 70)

    n_done    = sum(1 for r in results if r["status"] == "done")
    n_skipped = sum(1 for r in results if "skip" in r["status"])
    n_errors  = sum(1 for r in results if r["status"] == "error")
    n_empty   = sum(1 for r in results if r["status"] == "done_empty")
    total_rows = sum(r["n_rows"] for r in results)

    log(f"  Done      : {n_done}")
    log(f"  Skipped   : {n_skipped}  (already complete)")
    log(f"  Empty     : {n_empty}   (zero rows)")
    log(f"  Errors    : {n_errors}")
    log(f"  Total rows: {total_rows:,}")

    if n_done > 0:
        done_results = [r for r in results if r["status"] == "done"]
        avg_time = sum(r["elapsed"] for r in done_results) / n_done
        log(f"  Avg time/day: {_hms(avg_time)}")

    done_csvs = sorted(glob.glob(
        os.path.join(output_dir, "streak_trajectories_*_bright.csv")
    ))
    total_gb = sum(os.path.getsize(p) for p in done_csvs) / 1e9
    log(f"  Output files: {len(done_csvs)}  ({total_gb:.3f} GB total)")

    if n_errors > 0:
        log("")
        log("  [ERROR DAYS]:")
        for r in results:
            if r["status"] == "error":
                log(f"    {r['date']}")

    log(f"\n  Total wall time: {_hms(time.perf_counter() - wall_start)}")
    log("")
    log("[DONE]")
    log(f"[LOG]  {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(
        step2_dir    = "step2_output",           # same folder as script
        step3_dir    = "step3_output",
        sun_csv      = "step3_output/sun_positions.csv",
        output_dir   = "step4_output_20deg",
        power_kw     = 400.0,
        continuous   = False,
        n_workers    = 40,                        # match --cpus-per-task
        log_filename = "step4_20deg.log",
    )