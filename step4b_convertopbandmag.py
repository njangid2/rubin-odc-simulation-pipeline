"""
step4b_band_correction.py
─────────────────────────────────────────────────────────────────────────────
Corrects the step4 AB magnitude (computed at 532 nm) for each pointing's
actual LSST filter using solar colour offsets.

PROBLEM
───────
    step4 computes ab_magnitude assuming a single ~532 nm passband for every
    row, regardless of which LSST filter the pointing used.

    Each pointing has a pointing_filter (u/g/r/i/z/y).  The solar colour
    offset shifts the 532 nm magnitude to the correct band:

        ab_magnitude_corrected = ab_magnitude_532nm + solar_color[filter]

    Solar colours (Willmer 2018 / SMTN-002):
        u: +1.428   g: +0.245   r: -0.210
        i: -0.322   z: -0.357   y: -0.371

    NOTE: pointing_filter values in this dataset are suffixed
    (e.g. "r_57", "g_12") rather than bare single letters. The base band
    letter is extracted by splitting on "_" before the solar-color lookup —
    without this, every lookup misses and ab_magnitude_corrected comes out
    all-NaN for every row in every day.

INPUT FILES (per day, row-aligned)
───────────────────────────────────
    step2_output/streak_trajectories_YYYY-MM-DD.csv
        → columns used: pointing_filter   (one per row)

    step4_output_20deg/streak_trajectories_YYYY-MM-DD_bright.csv
        → columns used: ab_magnitude      (one per row, 532 nm)

OUTPUT FILE (per day, row-aligned, same length as input)
────────────────────────────────────────────────────────
    step4b_output/streak_trajectories_YYYY-MM-DD_bright_corrected.csv
        ab_magnitude_corrected   float32  — filter-corrected AB mag
        solar_color_applied      float32  — offset that was added
        pointing_filter          str      — band (for reference / sanity check)

    Row order and count are IDENTICAL to the step4 sidecar.
    Rows with no valid filter mapping (NaN ab_magnitude, unknown filter)
    get NaN in ab_magnitude_corrected.

RESUMABILITY
────────────
    Existing output files are skipped automatically.

PARALLELISM
───────────
    n_workers days processed simultaneously via multiprocessing.Pool.

USAGE
─────
    python step4b_band_correction.py
"""

import os
import gc
import glob
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# SOLAR COLOUR OFFSETS  (532 nm → LSST band)
# ═════════════════════════════════════════════════════════════════════════════

SOLAR_COLOR = {
    'u': +1.428,
    'g': +0.245,
    'r': -0.210,
    'i': -0.322,
    'z': -0.357,
    'y': -0.371,
}


def base_band(filter_value) -> str:
    """
    Extracts the base LSST band letter from a (possibly suffixed)
    pointing_filter value, e.g. "r_57" -> "r", "g_12" -> "g", "u" -> "u".
    """
    return str(filter_value).strip().split('_')[0]


# ═════════════════════════════════════════════════════════════════════════════
# WORKER — one day
# ═════════════════════════════════════════════════════════════════════════════

def process_one_day(args):
    """
    Applies per-row solar colour correction to one day's step4 sidecar.

    args = (date_str, step2_path, step4_path, output_path)
    """
    date_str, step2_path, step4_path, output_path = args
    t0 = time.perf_counter()

    # ── Resume ────────────────────────────────────────────────────────────
    if os.path.exists(output_path):
        return f"[SKIP] {date_str} — already exists"

    # ── Load pointing_filter from step2 ──────────────────────────────────
    try:
        s2 = pd.read_csv(step2_path, usecols=["pointing_filter"])
    except Exception as e:
        return f"[ERROR] {date_str} — step2 read failed: {e}"

    # ── Load ab_magnitude from step4 sidecar ─────────────────────────────
    try:
        s4 = pd.read_csv(step4_path, usecols=["ab_magnitude"])
    except Exception as e:
        return f"[ERROR] {date_str} — step4 read failed: {e}"

    N2, N4 = len(s2), len(s4)
    if N2 != N4:
        return (f"[ERROR] {date_str} — row mismatch: "
                f"step2={N2:,}  step4={N4:,}")

    # ── Per-row solar colour lookup ───────────────────────────────────────
    filters = s2["pointing_filter"].values          # string array, may be
                                                      # suffixed e.g. "r_57"
    ab_532  = s4["ab_magnitude"].values.astype(float)

    color_arr = np.array(
        [SOLAR_COLOR.get(base_band(f), np.nan) for f in filters],
        dtype=np.float32
    )

    n_unknown = int(np.isnan(color_arr).sum())
    if n_unknown:
        unknown_bands = set(
            base_band(f) for f, c in zip(filters, color_arr)
            if np.isnan(c)
        )
        print(f"  [WARN] {date_str} — {n_unknown:,} rows have unknown filter "
              f"(base band): {unknown_bands}", flush=True)

    # ── Apply correction ──────────────────────────────────────────────────
    ab_corrected = (ab_532 + color_arr).astype(np.float32)
    # Rows where ab_532 was already NaN stay NaN
    ab_corrected[np.isnan(ab_532)] = np.nan

    # ── Save (row-aligned sidecar) ────────────────────────────────────────
    out = pd.DataFrame({
        "pointing_filter":       filters,
        "solar_color_applied":   np.round(color_arr,   4).astype(np.float32),
        "ab_magnitude_corrected": np.round(ab_corrected, 6).astype(np.float32),
    })
    out.to_csv(output_path, index=False)

    n_valid = int(out["ab_magnitude_corrected"].notna().sum())

    del s2, s4, filters, ab_532, color_arr, ab_corrected, out
    gc.collect()

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(output_path) / 1e6

    return (f"[DONE] {date_str}  rows={N2:,}  "
            f"valid={n_valid:,}  {size_mb:.2f} MB  {elapsed:.1f}s")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(
    step2_dir  = "step2_output",          # streak_trajectories_YYYY-MM-DD.csv
    step4_dir  = "step4_output_20deg_sep",    # streak_trajectories_YYYY-MM-DD_bright.csv
    output_dir = "step4b_output",         # corrected sidecars go here
    n_workers  = 36,
):
    t_start = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] step2_dir  : {step2_dir}")
    print(f"[INFO] step4_dir  : {step4_dir}")
    print(f"[INFO] output_dir : {output_dir}")
    print(f"[INFO] n_workers  : {n_workers}")
    print(f"[INFO] solar colour offsets:")
    for b, sc in SOLAR_COLOR.items():
        print(f"         {b}: {sc:+.3f}")

    # ── Match step2 ↔ step4 by date ──────────────────────────────────────
    step4_paths = sorted(glob.glob(
        os.path.join(step4_dir, "streak_trajectories_*_bright.csv")
    ))
    if not step4_paths:
        raise FileNotFoundError(
            f"No streak_trajectories_*_bright.csv found in: {step4_dir}"
        )

    # Build date → step2 path map
    step2_map = {}
    for p in sorted(glob.glob(
        os.path.join(step2_dir, "streak_trajectories_*.csv")
    )):
        if "_azel" in p or "_bright" in p:
            continue
        date_str = (os.path.basename(p)
                    .replace("streak_trajectories_", "")
                    .replace(".csv", ""))
        step2_map[date_str] = p

    print(f"\n[INFO] {len(step4_paths)} step4 bright files found")
    print(f"[INFO] {len(step2_map)} step2 files mapped")

    # ── Sanity check: confirm at least one real (non-NaN-producing) filter
    #    value maps correctly, so a regression like this doesn't silently
    #    pass again. Peek at the first available step2 file.
    if step2_map:
        sample_path = next(iter(step2_map.values()))
        try:
            sample_filters = pd.read_csv(
                sample_path, usecols=["pointing_filter"], nrows=2000
            )["pointing_filter"].unique()
            sample_bands = sorted(set(base_band(f) for f in sample_filters))
            unmapped = [b for b in sample_bands if b not in SOLAR_COLOR]
            print(f"[INFO] Sample pointing_filter values: "
                  f"{list(sample_filters[:10])}")
            print(f"[INFO] Extracted base bands: {sample_bands}")
            if unmapped:
                print(f"[WARN] Base bands with NO solar-color mapping: "
                      f"{unmapped} — these rows will be NaN in every "
                      f"output file. Check base_band() / SOLAR_COLOR.")
        except Exception as e:
            print(f"[WARN] Could not sanity-check filter values: {e}")

    # ── Build task list ───────────────────────────────────────────────────
    tasks = []
    n_skip = 0
    for step4_path in step4_paths:
        date_str = (os.path.basename(step4_path)
                    .replace("streak_trajectories_", "")
                    .replace("_bright.csv", ""))

        output_path = os.path.join(
            output_dir,
            f"streak_trajectories_{date_str}_bright_corrected.csv"
        )

        if os.path.exists(output_path):
            n_skip += 1
            continue

        if date_str not in step2_map:
            print(f"  [WARN] No step2 for {date_str} — skipping", flush=True)
            continue

        tasks.append((date_str, step2_map[date_str], step4_path, output_path))

    print(f"[INFO] {n_skip} days already done — skipping")
    print(f"[INFO] {len(tasks)} days to process\n")

    if not tasks:
        print("[INFO] Nothing to do — all output files already exist.")
        return

    # ── Run parallel ──────────────────────────────────────────────────────
    n = min(n_workers, cpu_count(), len(tasks))
    print(f"[INFO] Launching {n} parallel workers …\n")

    with Pool(processes=n) as pool:
        for msg in pool.imap_unordered(process_one_day, tasks):
            print(msg, flush=True)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed  = time.perf_counter() - t_start
    all_out  = sorted(glob.glob(
        os.path.join(output_dir, "streak_trajectories_*_bright_corrected.csv")
    ))
    total_gb = sum(os.path.getsize(p) for p in all_out) / 1e9

    print(f"\n{'='*60}")
    print(f"[ALL DONE]  {len(all_out)} files  {total_gb:.2f} GB  {elapsed:.1f}s")
    print(f"{'='*60}")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main(
        step2_dir  = "step2_output",
        step4_dir  = "step4_output_20deg_rad",
        output_dir = "step4b_output_rad",
        n_workers  = 94,
    )