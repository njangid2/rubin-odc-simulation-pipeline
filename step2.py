"""
step2_propagate_and_match.py
─────────────────────────────
Replaces step2a + step2b with a single parallel pass:

  For each day:
    dispatch all shells for that day to N_WORKERS parallel workers
    each worker: load TLEs → propagate → FOV-match → save intermediate CSV
                 (position arrays never touch disk)
    main process: merge all shell intermediates → streak_trajectories_YYYY-MM-DD.csv
                  delete intermediates
    move to next day

PARALLELISM
───────────
Shells within each day run in parallel (Pool.map over shells).
Days run sequentially so the per-day merge + cleanup stays simple.
N_WORKERS = 8  (leaves 2 of your 10 cores free).

RESUMABILITY (two levels)
─────────────────────────
  Level 1 — streak_trajectories_YYYY-MM-DD.csv exists → skip entire day
  Level 2 — trajectories_shell_XXX_YYYY-MM-DD.csv exists → skip that shell

  Crash on day 4 shell 10 leaves:
      streak_trajectories_2025-11-01.csv  ✓ complete
      streak_trajectories_2025-11-02.csv  ✓ complete
      streak_trajectories_2025-11-03.csv  ✓ complete
      trajectories_shell_001_2025-11-04.csv  ← shells 1-9 done
      ...
      trajectories_shell_009_2025-11-04.csv
  Restart: days 1-3 skipped, day 4 resumes from shell 10.

LOGGING
───────
All output → terminal AND step2_output/step2.log simultaneously.
Each line timestamped + PID tagged.

    tail -f step2_output/step2.log          # watch live
    grep "MERGED" step2_output/step2.log    # completed days
    grep "WARN\|ERROR" step2_output/step2.log  # problems

STORAGE
───────
  - NO pos_cache pkl files written (was ~50 GB/day — eliminated)
  - Intermediates deleted after each day's merge
  - Only final streak_trajectories_YYYY-MM-DD.csv kept

Requirements:
    pip install sgp4
"""

import os, gc, time, pickle, math, glob, logging, sys
from multiprocessing import Pool
import functools
import numpy as np
import pandas as pd
import astropy.time
from sgp4.api import Satrec

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str, log_filename: str = "step2.log") -> str:
    """
    Configure root logger → terminal + log file simultaneously.
    Called once in main() and once at the start of each worker process.
    Returns full path to log file.
    """
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
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────
_TIMINGS: dict = {}

def _hms(s):
    h, r = divmod(int(s), 3600); m, sec = divmod(r, 60); frac = s - int(s)
    if h:  return f"{h}h {m}m {sec+frac:.2f}s"
    if m:  return f"{m}m {sec+frac:.2f}s"
    return f"{sec+frac:.2f}s"

class Timer:
    def __init__(self, label): self.label = label
    def __enter__(self): self.t0 = time.perf_counter(); return self
    def __exit__(self, *_):
        elapsed = time.perf_counter() - self.t0
        _TIMINGS[self.label] = _TIMINGS.get(self.label, 0.0) + elapsed
        log(f"  ⏱  {self.label}: {_hms(elapsed)}")

def log_timing_summary():
    if not _TIMINGS: return
    total = sum(_TIMINGS.values()); w = max(len(k) for k in _TIMINGS) + 2
    log("=" * 62)
    log("  TIMING SUMMARY")
    log("=" * 62)
    for label, elapsed in _TIMINGS.items():
        pct = 100.0 * elapsed / total if total else 0
        log(f"  {label:<{w}} {_hms(elapsed):>12}   ({pct:5.1f}%)")
    log("-" * 62)
    log(f"  {'TOTAL':<{w}} {_hms(total):>12}")
    log("=" * 62)

# ─────────────────────────────────────────────────────────────────────────────
# Column constants
# ─────────────────────────────────────────────────────────────────────────────
COL_ID      = "observationId"
COL_RA      = "fieldRA"
COL_DEC     = "fieldDec"
COL_MJD     = "observationStartMJD"
COL_EXPTIME = "visitExposureTime"
COL_FILTER  = "filter"
COL_NIGHT   = "night"

OUTPUT_COLS = [
    "pointing_id", "pointing_ra", "pointing_dec",
    "pointing_mjd", "pointing_exptime",
    "pointing_filter", "pointing_night",
    "shell_id", "sat_name", "step", "t_mjd",
    "ra_deg", "dec_deg",
    "sep_fov_center_deg", "in_fov", "sunlit",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mjd_noon(mjd: float) -> float:
    return math.floor(mjd) + 0.5

def mjd_to_date_str(mjd: float) -> str:
    return astropy.time.Time(mjd, format="mjd", scale="utc").iso[:10]

def load_tle_pkl(pkl_path: str) -> list:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["tle_lines"] if isinstance(data, dict) else data

def mjd_to_jd_split(mjd: np.ndarray):
    jd_full = mjd + 2400000.5
    jd1     = np.floor(jd_full).astype(np.float64)
    jd2     = (jd_full - jd1).astype(np.float64)
    return jd1, jd2

def build_mjd_grid(hit_day_df: pd.DataFrame, n_steps: int) -> np.ndarray:
    """Sorted unique MJD steps covering every hit pointing's exposure window."""
    mjds = set()
    for _, row in hit_day_df.iterrows():
        t0 = float(row[COL_MJD])
        t1 = t0 + float(row[COL_EXPTIME]) / 86400.0
        for t in np.linspace(t0, t1, n_steps):
            mjds.add(round(t, 9))
    return np.array(sorted(mjds))

# ─────────────────────────────────────────────────────────────────────────────
# TEME → GCRS → RA/Dec + sunlit
# ─────────────────────────────────────────────────────────────────────────────

def _gmst_rad(jd_ut1: np.ndarray) -> np.ndarray:
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = (67310.54841
                + (876600.0 * 3600.0 + 8640184.812866) * T
                + 0.093104 * T**2
                - 6.2e-6   * T**3)
    return np.radians(gmst_sec % 86400.0 / 240.0)

def teme_to_gcrs(pos_teme: np.ndarray, jd_ut1: np.ndarray) -> np.ndarray:
    theta  = _gmst_rad(jd_ut1)
    ct, st = np.cos(theta), np.sin(theta)
    return np.vstack([
         ct * pos_teme[0] + st * pos_teme[1],
        -st * pos_teme[0] + ct * pos_teme[1],
         pos_teme[2],
    ])

def gcrs_to_radec(pos_gcrs: np.ndarray) -> tuple:
    x, y, z = pos_gcrs
    r   = np.sqrt(x**2 + y**2 + z**2)
    ra  = (np.degrees(np.arctan2(y, x)) % 360.0).astype(np.float32)
    dec = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0))).astype(np.float32)
    return ra, dec

def compute_sunlit(pos_gcrs: np.ndarray, jd_full: np.ndarray) -> np.ndarray:
    RE   = 6378.137
    n_jd = jd_full - 2451545.0
    L    = np.radians((280.460 + 0.9856474 * n_jd) % 360)
    g    = np.radians((357.528 + 0.9856003 * n_jd) % 360)
    lam  = L + np.radians(1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps  = np.radians(23.439 - 4e-7 * n_jd)
    sun  = np.vstack([np.cos(lam),
                      np.cos(eps) * np.sin(lam),
                      np.sin(eps) * np.sin(lam)])
    proj = -np.einsum('it,it->t', sun, pos_gcrs)
    r2   =  np.einsum('it,it->t', pos_gcrs, pos_gcrs)
    return (proj < 0) | (r2 - proj**2 > RE**2)

def _angular_sep_deg(ra1: float, dec1: float,
                     ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    r1  = np.radians(ra1);  d1 = np.radians(dec1)
    r2  = np.radians(ra2);  d2 = np.radians(dec2)
    cos = (np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(r1-r2))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

# ─────────────────────────────────────────────────────────────────────────────
# CORE: propagate + FOV-match in one pass  (positions never touch disk)
# ─────────────────────────────────────────────────────────────────────────────

def propagate_and_match(
    tle_lines:      list,
    hit_day_df:     pd.DataFrame,
    shell_id:       int,
    n_steps:        int   = 10,
    fov_radius_deg: float = 1.75,
    report_every:   int   = 1000,
) -> pd.DataFrame:
    """
    Propagate all satellites and FOV-match against hit pointings in one pass.
    Position arrays live in RAM only — freed before returning.
    Returns DataFrame of matched rows, empty if no FOV hits.
    """
    if hit_day_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    mjd_grid = build_mjd_grid(hit_day_df, n_steps=n_steps)
    T        = len(mjd_grid)
    S        = len(tle_lines)

    jd1, jd2 = mjd_to_jd_split(mjd_grid)
    jd_full   = jd1 + jd2

    log(f"    S{shell_id:03d} | time grid: {T:,} steps  "
        f"sats: {S:,}  "
        f"peak RAM ≈ {2*T*S*4/1e6:.0f} MB")

    # ── Propagate ─────────────────────────────────────────────────────────
    ra_all  = np.full((T, S), np.nan, dtype=np.float32)
    dec_all = np.full((T, S), np.nan, dtype=np.float32)
    sun_all = np.zeros((T, S), dtype=bool)
    sat_names = []
    n_failed  = 0

    for s_idx, tle in enumerate(tle_lines):
        parts = tle.strip().split("\n")
        if len(parts) != 3:
            sat_names.append(f"PARSE_ERR_{s_idx}")
            n_failed += 1
            continue

        name, l1, l2 = parts
        sat_names.append(name)

        try:
            sat = Satrec.twoline2rv(l1, l2)
        except Exception:
            n_failed += 1
            continue

        e, r, _ = sat.sgp4_array(jd1, jd2)
        ok = (e == 0)
        if not ok.any():
            n_failed += 1
            continue

        pos_teme           = r.T.astype(np.float64)
        pos_teme[:, ~ok]   = 0.0
        pos_gcrs           = teme_to_gcrs(pos_teme, jd_full)
        ra, dec            = gcrs_to_radec(pos_gcrs)
        ra_all [ok, s_idx] = ra [ok]
        dec_all[ok, s_idx] = dec[ok]
        sun_all[ok, s_idx] = compute_sunlit(pos_gcrs, jd_full)[ok]

        if (s_idx + 1) % report_every == 0 or s_idx == S - 1:
            log(f"    S{shell_id:03d} | propagated {s_idx+1:,}/{S:,} sats")

    if n_failed:
        log(f"    S{shell_id:03d} | [WARN] {n_failed}/{S} sats failed propagation")

    # ── FOV matching ───────────────────────────────────────────────────────
    sat_names_arr = np.array(sat_names)
    rows          = []
    n_pointings   = len(hit_day_df)

    for p_idx, (_, row) in enumerate(hit_day_df.iterrows()):
        pid      = int(row[COL_ID])
        pt_ra    = float(row[COL_RA])
        pt_dec   = float(row[COL_DEC])
        pt_mjd   = float(row[COL_MJD])
        pt_exp   = float(row[COL_EXPTIME])
        pt_filt  = row[COL_FILTER]
        pt_night = int(row[COL_NIGHT])
        pt_end   = pt_mjd + pt_exp / 86400.0

        t_mask = (mjd_grid >= pt_mjd) & (mjd_grid <= pt_end)
        t_idxs = np.where(t_mask)[0]
        if len(t_idxs) == 0:
            continue

        sep_matrix = np.empty((len(t_idxs), S), dtype=np.float32)
        for i, t_idx in enumerate(t_idxs):
            sep_matrix[i] = _angular_sep_deg(
                pt_ra, pt_dec, ra_all[t_idx], dec_all[t_idx]
            )

        hit_sat_idxs = np.where(
            np.any(sep_matrix <= fov_radius_deg, axis=0)
        )[0]
        if len(hit_sat_idxs) == 0:
            continue

        for s_idx in hit_sat_idxs:
            for step, t_idx in enumerate(t_idxs):
                sep = float(sep_matrix[step, s_idx])
                rows.append({
                    "pointing_id":        pid,
                    "pointing_ra":        pt_ra,
                    "pointing_dec":       pt_dec,
                    "pointing_mjd":       pt_mjd,
                    "pointing_exptime":   pt_exp,
                    "pointing_filter":    pt_filt,
                    "pointing_night":     pt_night,
                    "shell_id":           shell_id,
                    "sat_name":           sat_names_arr[s_idx],
                    "step":               step,
                    "t_mjd":              round(float(mjd_grid[t_idx]), 9),
                    "ra_deg":             round(float(ra_all [t_idx, s_idx]), 5),
                    "dec_deg":            round(float(dec_all[t_idx, s_idx]), 5),
                    "sep_fov_center_deg": round(sep, 5),
                    "in_fov":             bool(sep <= fov_radius_deg),
                    "sunlit":             bool(sun_all[t_idx, s_idx]),
                })

        if (p_idx + 1) % 500 == 0 or p_idx == n_pointings - 1:
            log(f"    S{shell_id:03d} | matched {p_idx+1:,}/{n_pointings:,} "
                f"pointings  rows so far: {len(rows):,}")

    # Free position arrays immediately
    del ra_all, dec_all, sun_all
    gc.collect()

    return (pd.DataFrame(rows, columns=OUTPUT_COLS)
            if rows else pd.DataFrame(columns=OUTPUT_COLS))

# ─────────────────────────────────────────────────────────────────────────────
# Worker function — processes ONE shell × ONE day
# Must be top-level for Mac multiprocessing spawn to pickle it
# ─────────────────────────────────────────────────────────────────────────────

def process_shell_day(shell_id, date_str, step1_dir, output_dir,
                      n_steps, fov_radius_deg, log_path):
    """
    Worker: propagate + FOV-match one shell × one day.
    Saves result to intermediate CSV.
    Returns (shell_id, date_str, n_rows, size_mb) for main process logging.
    """
    # Re-init logging in this worker (each spawned process starts fresh)
    setup_logging(output_dir, os.path.basename(log_path))

    shell_tag = f"shell_{shell_id:03d}"
    inter_csv = os.path.join(
        output_dir, f"trajectories_{shell_tag}_{date_str}.csv"
    )

    # Resume check
    if os.path.isfile(inter_csv):
        log(f"[SKIP] {os.path.basename(inter_csv)} already exists.")
        size_mb = os.path.getsize(inter_csv) / 1e6
        try:
            n_rows = len(pd.read_csv(inter_csv))
        except Exception:
            n_rows = 0
        return shell_id, date_str, n_rows, size_mb, True  # True = was skipped

    t_start = time.perf_counter()
    log(f"[START] Shell {shell_id:03d} × {date_str}")

    # Load hit pointings for this shell × day
    streak_csv = os.path.join(step1_dir, f"streak_results_{shell_tag}.csv")
    if not os.path.isfile(streak_csv):
        log(f"[WARN]  Shell {shell_id:03d} — streak CSV missing: {streak_csv}")
        pd.DataFrame(columns=OUTPUT_COLS).to_csv(inter_csv, index=False)
        return shell_id, date_str, 0, 0.0, False

    df         = pd.read_csv(streak_csv)
    hit_df     = df[df["n_streaks"] > 0].copy()
    hit_df["date_str"] = hit_df[COL_MJD].apply(mjd_to_date_str)
    day_hit_df = hit_df[hit_df["date_str"] == date_str].copy()
    del df, hit_df
    gc.collect()

    log(f"  S{shell_id:03d} × {date_str} | hit pointings: {len(day_hit_df):,}")

    if day_hit_df.empty:
        pd.DataFrame(columns=OUTPUT_COLS).to_csv(inter_csv, index=False)
        log(f"  S{shell_id:03d} × {date_str} | no hit pointings — empty CSV written.")
        return shell_id, date_str, 0, 0.0, False

    # Load TLEs
    tle_pkl = os.path.join(
        step1_dir, f"tles_shell{shell_id:03d}_{date_str}.pkl"
    )
    if not os.path.isfile(tle_pkl):
        log(f"[WARN]  Shell {shell_id:03d} × {date_str} — TLE pkl missing: {tle_pkl}")
        pd.DataFrame(columns=OUTPUT_COLS).to_csv(inter_csv, index=False)
        return shell_id, date_str, 0, 0.0, False

    t0        = time.perf_counter()
    tle_lines = load_tle_pkl(tle_pkl)
    log(f"  S{shell_id:03d} × {date_str} | {len(tle_lines):,} TLEs loaded "
        f"({time.perf_counter()-t0:.2f}s)")

    # Propagate + match (positions stay in RAM only)
    t0        = time.perf_counter()
    result_df = propagate_and_match(
        tle_lines      = tle_lines,
        hit_day_df     = day_hit_df,
        shell_id       = shell_id,
        n_steps        = n_steps,
        fov_radius_deg = fov_radius_deg,
    )
    prop_time = time.perf_counter() - t0

    del tle_lines, day_hit_df
    gc.collect()

    # Save intermediate CSV
    result_df.to_csv(inter_csv, index=False)
    size_mb = os.path.getsize(inter_csv) / 1e6
    n_rows  = len(result_df)
    n_sats  = result_df["sat_name"].nunique() if not result_df.empty else 0

    total_time = time.perf_counter() - t_start
    if result_df.empty:
        log(f"  S{shell_id:03d} × {date_str} | no FOV hits  "
            f"prop={_hms(prop_time)}  total={_hms(total_time)}")
    else:
        log(f"  S{shell_id:03d} × {date_str} | "
            f"rows={n_rows:,}  sats={n_sats:,}  "
            f"{size_mb:.2f} MB  "
            f"prop={_hms(prop_time)}  total={_hms(total_time)}")

    del result_df
    gc.collect()

    return shell_id, date_str, n_rows, size_mb, False

# ─────────────────────────────────────────────────────────────────────────────
# Per-day merge + cleanup
# ─────────────────────────────────────────────────────────────────────────────

def merge_day_and_cleanup(date_str: str, shell_ids: list,
                          output_dir: str) -> bool:
    """
    Merge all shell intermediate CSVs for date_str → final CSV.
    Deletes intermediates ONLY after final CSV confirmed on disk.
    Returns True if successful.
    """
    final_path = os.path.join(
        output_dir, f"streak_trajectories_{date_str}.csv"
    )

    inter_paths = []
    for shell_id in shell_ids:
        p = os.path.join(
            output_dir,
            f"trajectories_shell_{shell_id:03d}_{date_str}.csv"
        )
        if os.path.isfile(p):
            inter_paths.append(p)

    if not inter_paths:
        log(f"[WARN]  No intermediates found for {date_str} — skipping merge.")
        return False

    frames = []
    for path in inter_paths:
        try:
            tmp = pd.read_csv(path)
            if not tmp.empty:
                frames.append(tmp)
        except Exception as e:
            log(f"[WARN]  Could not read {path}: {e}")

    day_df = (pd.concat(frames, ignore_index=True)
              if frames else pd.DataFrame(columns=OUTPUT_COLS))
    del frames
    gc.collect()

    day_df.to_csv(final_path, index=False)

    if not os.path.isfile(final_path):
        log(f"[ERROR] Final CSV missing after write: {final_path}")
        return False

    size_mb  = os.path.getsize(final_path) / 1e6
    n_shells = day_df["shell_id"].nunique() if not day_df.empty else 0
    n_sats   = day_df["sat_name"].nunique() if not day_df.empty else 0
    n_pts    = day_df["pointing_id"].nunique() if not day_df.empty else 0
    log(f"[MERGED] streak_trajectories_{date_str}.csv  "
        f"({size_mb:.1f} MB)  "
        f"{len(day_df):,} rows  |  "
        f"{n_shells} shell(s)  |  "
        f"{n_sats:,} sats  |  "
        f"{n_pts:,} pointings")

    del day_df
    gc.collect()

    # Delete intermediates now that final is confirmed
    n_deleted = 0
    for path in inter_paths:
        try:
            os.remove(path)
            n_deleted += 1
        except OSError as e:
            log(f"[WARN]  Could not delete {path}: {e}")
    log(f"[CLEAN]  Deleted {n_deleted}/{len(inter_paths)} intermediate CSV(s).")

    return True

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(
    step1_dir      = "streak_output_sgp4",
    output_dir     = "step2_output",
    shell_ids      = None,
    n_steps        = 10,
    fov_radius_deg = 1.75,
    n_workers      = 8,
    delete_tles    = False,
    log_filename   = "step2.log",
):
    # ── Set up logging (main process) ───────────────────────────────────────
    log_path = setup_logging(output_dir, log_filename)

    wall_start = time.perf_counter()
    _TIMINGS.clear()

    log("=" * 70)
    log("  STEP 2: Propagate + FOV Match  (parallel shells, per-day merge)")
    log("=" * 70)
    log(f"  step1_dir      : {step1_dir}")
    log(f"  output_dir     : {output_dir}")
    log(f"  n_steps        : {n_steps}")
    log(f"  fov_radius_deg : {fov_radius_deg}°")
    log(f"  n_workers      : {n_workers}")
    log(f"  delete_tles    : {delete_tles}")
    log(f"  log file       : {log_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Auto-detect shells ─────────────────────────────────────────────────
    if shell_ids is None:
        csvs = sorted(glob.glob(
            os.path.join(step1_dir, "streak_results_shell_*.csv")
        ))
        if not csvs:
            raise FileNotFoundError(
                f"No streak_results_shell_*.csv in '{step1_dir}'. "
                "Run step1 first."
            )
        shell_ids = [
            int(os.path.basename(p)
                .replace("streak_results_shell_", "")
                .replace(".csv", ""))
            for p in csvs
        ]
        log(f"[AUTO] Detected {len(shell_ids)} shell(s): {shell_ids}")

    # ── Scan which dates each shell has hits on ────────────────────────────
    log("")
    log("[INFO] Scanning hit dates across all shells …")
    all_dates_per_shell: dict = {}

    for shell_id in shell_ids:
        shell_tag  = f"shell_{shell_id:03d}"
        streak_csv = os.path.join(step1_dir, f"streak_results_{shell_tag}.csv")
        if not os.path.isfile(streak_csv):
            log(f"[WARN]  {streak_csv} not found — shell {shell_id} skipped.")
            continue
        df     = pd.read_csv(streak_csv, usecols=[COL_MJD, "n_streaks"])
        hit_df = df[df["n_streaks"] > 0]
        all_dates_per_shell[shell_id] = set(
            hit_df[COL_MJD].apply(mjd_to_date_str)
        )
        del df, hit_df
        gc.collect()

    if not all_dates_per_shell:
        log("[WARN] No shells with hit pointings found. Exiting.")
        return

    all_dates = sorted(set().union(*all_dates_per_shell.values()))
    log(f"[INFO] {len(all_dates)} unique date(s): "
        f"{all_dates[0]} … {all_dates[-1]}")

    # ── Outer day loop ─────────────────────────────────────────────────────
    for day_idx, date_str in enumerate(all_dates):

        log("")
        log("=" * 70)
        log(f"  DAY {day_idx+1}/{len(all_dates)}  :  {date_str}")
        log("=" * 70)

        final_path = os.path.join(
            output_dir, f"streak_trajectories_{date_str}.csv"
        )

        # ── RESUME LEVEL 1: entire day already merged ──────────────────────
        if os.path.isfile(final_path):
            log(f"[SKIP] streak_trajectories_{date_str}.csv exists "
                f"— entire day complete.")
            continue

        # Shells that have hits on this date
        shells_this_day = [
            sid for sid in shell_ids
            if sid in all_dates_per_shell
            and date_str in all_dates_per_shell[sid]
        ]

        if not shells_this_day:
            log(f"[NOTE] No shells with hits on {date_str} — skipping.")
            continue

        log(f"[INFO] {len(shells_this_day)} shell(s) to process on {date_str}  "
            f"using {n_workers} workers")
        log(f"[INFO] Worker output interleaved below — use PID column to sort")

        # ── RESUME LEVEL 2: filter to shells not yet done ─────────────────
        shells_todo = []
        for sid in shells_this_day:
            inter_csv = os.path.join(
                output_dir,
                f"trajectories_shell_{sid:03d}_{date_str}.csv"
            )
            if os.path.isfile(inter_csv):
                log(f"[SKIP] trajectories_shell_{sid:03d}_{date_str}.csv exists.")
            else:
                shells_todo.append(sid)

        if not shells_todo:
            log(f"[INFO] All shells for {date_str} already processed — merging.")
        else:
            log(f"[INFO] {len(shells_todo)} shell(s) need processing: {shells_todo}")

            # Build worker args: one tuple per shell
            worker_args = [
                (sid, date_str, step1_dir, output_dir,
                 n_steps, fov_radius_deg, log_path)
                for sid in shells_todo
            ]

            # ── Dispatch shells in parallel ────────────────────────────────
            with Timer(f"parallel shells {date_str}"):
                with Pool(processes=min(n_workers, len(shells_todo))) as pool:
                    worker_results = pool.starmap(process_shell_day, worker_args)

            # Log summary of worker results
            log("")
            log(f"[SUMMARY] Day {date_str} worker results:")
            for sid, ds, n_rows, size_mb, was_skipped in worker_results:
                status = "SKIP" if was_skipped else "DONE"
                log(f"  [{status}] Shell {sid:03d}  rows={n_rows:,}  {size_mb:.2f} MB")

            # Optionally delete TLE pkls for this day
            if delete_tles:
                for sid in shells_todo:
                    tle_pkl = os.path.join(
                        step1_dir, f"tles_shell{sid:03d}_{date_str}.pkl"
                    )
                    if os.path.isfile(tle_pkl):
                        try:
                            os.remove(tle_pkl)
                            log(f"[DEL]  {os.path.basename(tle_pkl)}")
                        except OSError as e:
                            log(f"[WARN] Could not delete TLE pkl: {e}")

        # ── Merge all shell intermediates for this day ─────────────────────
        log("")
        log(f"── Merging day {date_str} ──")
        with Timer(f"merge+cleanup {date_str}"):
            merge_day_and_cleanup(
                date_str   = date_str,
                shell_ids  = shell_ids,
                output_dir = output_dir,
            )

    # ── Final summary ──────────────────────────────────────────────────────
    log("")
    log("=" * 70)
    final_csvs      = sorted(glob.glob(
        os.path.join(output_dir, "streak_trajectories_*.csv")))
    inter_remaining = sorted(glob.glob(
        os.path.join(output_dir, "trajectories_shell_*.csv")))
    final_gb = sum(os.path.getsize(p) for p in final_csvs) / 1e9
    tle_gb   = sum(
        os.path.getsize(p)
        for p in glob.glob(os.path.join(step1_dir, "tles_shell*.pkl"))
    ) / 1e9

    log(f"  Final date CSVs   : {len(final_csvs):>4} files   {final_gb:.2f} GB")
    log(f"  Intermediates left: {len(inter_remaining):>4} files   "
        f"(should be 0 if all days completed)")
    log(f"  TLE pkls remaining: "
        f"{len(glob.glob(os.path.join(step1_dir, 'tles_shell*.pkl'))):>4} files   "
        f"{tle_gb:.2f} GB")
    log(f"  pos_cache pkls    :    0 files   0.00 GB  ← eliminated")

    if inter_remaining:
        log(f"[NOTE] {len(inter_remaining)} intermediate(s) remain "
            f"— incomplete day detected. Re-run to finish.")

    _TIMINGS["TOTAL WALL TIME"] = time.perf_counter() - wall_start
    log("")
    log_timing_summary()
    log("")
    log("[DONE]")
    log(f"[LOG]  Full log saved to: {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(
        step1_dir      = "streak_output_sgp4",
        output_dir     = "step2_output",
        shell_ids      = None,        # None → auto-detect all shells
        n_steps        = 10,          # time samples per exposure
        fov_radius_deg = 1.75,
        n_workers      = 40,           # 8 of your 10 logical cores
        delete_tles    = False,       # True → delete TLE pkls after each day
        log_filename   = "step2.log",
    )
