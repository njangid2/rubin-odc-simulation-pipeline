"""
step6_streak_brightness.py
─────────────────────────────────────────────────────────────────────────────
Computes streak surface brightness (mag/arcsec²) per day per band.

APPROACH: one file at a time
─────────────────────────────
For each step5 file (one day):
    1. Load step5 file
    2. Load matching step2 file → compute ang_vel + elevation
    3. Compute brightness for each band (parallel, 6 workers)
    4. Save one CSV per band → step6_output/step6_{date}_{band}.csv
    5. Free memory → move to next day

Multiple days processed simultaneously via outer Pool.
No nested pools — flat architecture avoids daemonic process error.

PARALLELISM
───────────
    n_workers days processed simultaneously.
    Each day spawns NO child processes — bands processed sequentially
    within the day worker to avoid nesting.
    For band-level parallelism across days, use n_workers=36 (one per
    day×band task if you have enough cores).

    Recommended for 40 cores:
        n_workers = 36  (36 day-band tasks simultaneously)
        Each task is one band on one day — small memory footprint.

RESUMABILITY
────────────
    Already-existing output files are skipped automatically.

REFERENCES
──────────
    SMTN-002 : https://smtn-002.lsst.io
    PSTN-054 : https://pstn-054.lsst.io  doi:10.71929/rubin/2584366
    syseng_throughputs v1.9: https://github.com/lsst-pst/syseng_throughputs

USAGE
─────
    python step6_streak_brightness.py
"""

import os, glob, gc, math, time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import lumos.constants as _lumos_const

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

PLATE_SCALE_ARCSEC_PX = 0.2
PSF_FWHM_ARCSEC       = 0.67
PIXEL_AREA_ARCSEC2    = PLATE_SCALE_ARCSEC_PX ** 2
D_MIRROR_M            = 8.36
A_MIRROR_M2           = math.pi * (6.49 / 2) ** 2
H_PLANCK              = 6.626e-34
AB_ZP_W               = 3631e-26
_LUMOS_C              = _lumos_const.SPEED_OF_LIGHT

BAND_PARAMS = {
    'u': dict(zp=26.52, tp=0.1397, dlam_m=5.587e-08, lam_m=366e-9, solar_color=+1.428),
    'g': dict(zp=28.51, tp=0.4592, dlam_m=1.424e-07, lam_m=482e-9, solar_color=+0.245),
    'r': dict(zp=28.36, tp=0.5361, dlam_m=1.405e-07, lam_m=622e-9, solar_color=-0.210),
    'i': dict(zp=28.17, tp=0.6224, dlam_m=1.248e-07, lam_m=754e-9, solar_color=-0.322),
    'z': dict(zp=27.78, tp=0.6134, dlam_m=1.021e-07, lam_m=869e-9, solar_color=-0.357),
    'y': dict(zp=26.82, tp=0.3180, dlam_m=9.015e-08, lam_m=971e-9, solar_color=-0.371),
}

POINT_SOURCE_DEPTH = {
    'u': 23.8, 'g': 24.5, 'r': 24.0,
    'i': 23.4, 'z': 22.7, 'y': 22.0,
}

SHELL_ALT_KM = {
     1: 686.0,   2: 687.3,   3: 688.7,   4: 690.0,   5: 691.3,
     6: 692.7,   7: 694.0,   8: 695.3,   9: 696.7,  10: 698.0,
    11: 699.3,  12: 700.7,  13: 702.0,  14: 703.3,  15: 704.7,
    16: 706.0,  17: 707.3,  18: 708.7,  19: 710.0,  20: 711.3,
    21: 712.7,  22: 714.0,  23: 715.3,  24: 716.7,  25: 718.0,
    26: 946.0,  27: 947.3,  28: 948.7,  29: 950.0,  30: 951.3,
    31: 952.7,  32: 954.0,  33: 955.3,  34: 956.7,  35: 958.0,
    36: 959.3,  37: 960.7,  38: 962.0,  39: 963.3,  40: 964.7,
    41: 966.0,  42: 967.3,  43: 968.7,  44: 970.0,  45: 971.3,
    46: 972.7,  47: 974.0,  48: 975.3,  49: 976.7,  50: 978.0,
    51: 707.0,  52: 708.8,  53: 710.5,  54: 712.3,  55: 714.0,
    56: 715.8,  57: 717.6,  58: 719.3,  59: 721.1,  60: 722.9,
    61: 724.6,  62: 726.4,  63: 728.1,  64: 729.9,  65: 731.7,
    66: 733.4,  67: 735.2,  68: 737.0,  69: 738.7,  70: 740.5,
    71: 742.2,  72: 744.0,
    73: 967.0,  74: 968.7,  75: 970.3,  76: 972.0,  77: 973.7,
    78: 975.3,  79: 977.0,  80: 978.7,  81: 980.3,  82: 982.0,
    83: 983.7,  84: 985.3,  85: 987.0,  86: 988.7,  87: 990.3,
    88: 992.0,  89: 993.7,  90: 995.3,  91: 997.0,  92: 998.7,
    93: 1000.3, 94: 1002.0,
}

# ═════════════════════════════════════════════════════════════════════════════
# PHYSICS
# ═════════════════════════════════════════════════════════════════════════════

def effective_psf_arcsec(sat_altitude_km, elevation_deg):
    alt_m         = np.asarray(sat_altitude_km, dtype=float) * 1000.0
    el_rad        = np.radians(np.clip(elevation_deg, 1.0, 90.0))
    slant_m       = alt_m / np.sin(el_rad)
    theta_defocus = 206265.0 * D_MIRROR_M / slant_m
    theta_eff     = np.sqrt(PSF_FWHM_ARCSEC**2 + theta_defocus**2)
    return theta_defocus, theta_eff


def ang_sep_arcsec(ra1, dec1, ra2, dec2):
    r1 = np.radians(ra1);  d1 = np.radians(dec1)
    r2 = np.radians(ra2);  d2 = np.radians(dec2)
    dra  = r1 - r2;  ddec = d1 - d2
    a = np.sin(ddec/2)**2 + np.cos(d1)*np.cos(d2)*np.sin(dra/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))) * 3600.0


# ═════════════════════════════════════════════════════════════════════════════
# MAIN WORKER — one day, all bands, one file at a time
# Top-level for multiprocessing pickle
# ═════════════════════════════════════════════════════════════════════════════

def process_one_day(args):
    """
    Processes ONE step5 file completely:
      1. Load step5 (one day)
      2. Load step2 (same day) → angular velocity + elevation
      3. For each band in this day's data:
           - compute brightness
           - save step6_{date}_{band}.csv
           - free band dataframe
      4. Free all memory

    No child pools — all bands processed sequentially within this worker.
    Multiple days run in parallel via the outer Pool.

    args = (date_str, step5_path, step2_path, step3_dir, output_dir)
    """
    date_str, step5_path, step2_path, step3_dir, output_dir = args
    t0 = time.perf_counter()

    # ── Skip if all bands done ────────────────────────────────────────────
    all_done = all(
        os.path.exists(os.path.join(output_dir, f"step6_{date_str}_{b}.csv"))
        for b in BAND_PARAMS
    )
    if all_done:
        return f"[SKIP] {date_str}"

    # ── Load step5 ────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(step5_path)
    except Exception as e:
        return f"[ERROR] {date_str} step5: {e}"

    if df.empty:
        return f"[SKIP] {date_str} empty"

    if "shell_alt_km" not in df.columns:
        df["shell_alt_km"] = df["shell_id"].map(SHELL_ALT_KM)

    # ── Load step2 ────────────────────────────────────────────────────────
    try:
        s2 = pd.read_csv(step2_path, usecols=[
            "pointing_id", "sat_name", "shell_id",
            "t_mjd", "ra_deg", "dec_deg", "in_fov"
        ])
    except Exception as e:
        return f"[ERROR] {date_str} step2: {e}"

    # ── Load step3 azel ───────────────────────────────────────────────────
    azel_path = os.path.join(
        step3_dir, f"streak_trajectories_{date_str}_azel.csv"
    )
    if os.path.exists(azel_path):
        try:
            azel = pd.read_csv(azel_path, usecols=["el_deg"])
            s2["el_deg"] = azel["el_deg"].values
            del azel
        except Exception:
            s2["el_deg"] = np.nan
    else:
        s2["el_deg"] = np.nan

    # ── Angular velocity + elevation per streak ───────────────────────────
    s2_infov = s2[s2["in_fov"] == True].copy()
    s2.sort_values(["pointing_id", "sat_name", "shell_id", "t_mjd"], inplace=True)
    s2_infov.sort_values(["pointing_id", "sat_name", "shell_id", "t_mjd"], inplace=True)

    def streak_stats(grp):
        if len(grp) >= 2:
            ra  = grp["ra_deg"].values
            dec = grp["dec_deg"].values
            t   = grp["t_mjd"].values * 86400.0
            seps      = ang_sep_arcsec(ra[:-1], dec[:-1], ra[1:], dec[1:])
            total_arc = seps.sum()
            total_dt  = t[-1] - t[0]
            if total_dt > 0 and total_arc > 0:
                el_vals = grp["el_deg"].values
                med_el  = float(np.nanmedian(el_vals)) if not np.all(np.isnan(el_vals)) else np.nan
                return pd.Series({
                    "ang_vel_arcsec_per_s": total_arc / total_dt,
                    "t_crossing_sec":       total_dt,
                    "median_elevation_deg": med_el,
                })

        # 1-point streaks: cannot reliably compute brightness — skip
        return pd.Series({
            "ang_vel_arcsec_per_s": np.nan,
            "t_crossing_sec":       np.nan,
            "median_elevation_deg": np.nan,
        })

    vel_df = (s2_infov
              .groupby(["pointing_id", "sat_name", "shell_id"], sort=False)
              .apply(streak_stats)
              .reset_index())

    del s2_infov; gc.collect()

    df = df.merge(
        vel_df[["pointing_id", "sat_name", "shell_id",
                "ang_vel_arcsec_per_s", "t_crossing_sec",
                "median_elevation_deg"]],
        on=["pointing_id", "sat_name", "shell_id"],
        how="left"
    )

    del s2, vel_df; gc.collect()

    # ── Process each band, save, free ─────────────────────────────────────
    msgs = []
    for band, band_df in df.groupby("pointing_filter"):
        if band not in BAND_PARAMS:
            continue

        out_path = os.path.join(output_dir, f"step6_{date_str}_{band}.csv")
        if os.path.exists(out_path):
            msgs.append(f"  [{band}] SKIP")
            continue

        bp          = BAND_PARAMS[band]
        ZP          = bp['zp'];   tp     = bp['tp']
        dlam_m      = bp['dlam_m']; lam_m  = bp['lam_m']
        solar_color = bp['solar_color']
        depth_limit = POINT_SOURCE_DEPTH[band]

        ab_mag          = band_df["ab_magnitude"].values.astype(float)
        L_px            = band_df["L_px"].values.astype(float)
        t_crossing_sec  = band_df["t_crossing_sec"].values.astype(float)
        sat_altitude_km = band_df["shell_alt_km"].values.astype(float)
        elevation_deg   = band_df["median_elevation_deg"].values.astype(float)

        ab_mag_band = ab_mag + solar_color
        detectable  = ab_mag_band < depth_limit

        theta_defocus, theta_eff = effective_psf_arcsec(sat_altitude_km, elevation_deg)
        psf_eff_px = theta_eff / PLATE_SCALE_ARCSEC_PX

        valid = (
            ~np.isnan(ab_mag_band) & ~np.isnan(L_px) &
            ~np.isnan(t_crossing_sec) & ~np.isnan(sat_altitude_km) &
            ~np.isnan(elevation_deg) &
            (L_px > 0) & (t_crossing_sec > 0) &
            (sat_altitude_km > 0) & (elevation_deg > 0)
        )

        e_per_px = np.full(len(ab_mag), np.nan)
        sb        = np.full(len(ab_mag), np.nan)

        if valid.any():
            m   = ab_mag_band[valid]; L  = L_px[valid]
            tc  = t_crossing_sec[valid]; psf = psf_eff_px[valid]

            F_nu      = 10.0**(-m / 2.5) * AB_ZP_W
            F_lam     = F_nu * _LUMOS_C / lam_m**2
            intensity = F_lam * dlam_m
            E_photon  = H_PLANCK * _LUMOS_C / lam_m
            phot_flux = intensity / E_photon

            N_photons       = phot_flux * A_MIRROR_M2 * tp * tc * 0.954
            streak_width_px = 1.698 * psf
            ep              = N_photons / (L * streak_width_px)
            e_per_px[valid] = ep

            flux_per_arcsec2 = (ep / tc) / PIXEL_AREA_ARCSEC2
            good    = flux_per_arcsec2 > 0
            sb_vals = np.full(len(ep), np.nan)
            sb_vals[good] = ZP - 2.5 * np.log10(flux_per_arcsec2[good])
            sb[valid] = sb_vals

        # Assemble and save immediately
        out = band_df.copy()
        out["ab_magnitude_band"]          = ab_mag_band
        out["solar_color_applied"]        = solar_color
        out["detectable"]                 = detectable
        out["theta_defocus_arcsec"]       = np.round(theta_defocus, 4)
        out["theta_eff_arcsec"]           = np.round(theta_eff, 4)
        out["streak_brightness_e_per_px"] = e_per_px
        out["surface_brightness_mag_arcsec2"] = np.where(sb > 24.5, np.nan, sb)

        out_cols = [
            "pointing_id", "sat_name", "shell_id",
            "pointing_filter", "pointing_exptime",
            "ab_magnitude", "ab_magnitude_band", "solar_color_applied",
            "detectable", "L_px", "shell_alt_km",
            "ang_vel_arcsec_per_s", "t_crossing_sec",
            "median_elevation_deg",
            "theta_defocus_arcsec", "theta_eff_arcsec",
            "peak_electrons",
            "streak_brightness_e_per_px",
            "surface_brightness_mag_arcsec2",
        ]
        out_cols = [c for c in out_cols if c in out.columns]
        out[out_cols].to_csv(out_path, index=False)

        n_valid      = int(np.sum(~np.isnan(e_per_px)))
        n_detectable = int(np.sum(detectable & ~np.isnan(ab_mag_band)))
        msgs.append(
            f"  [{band}] {len(band_df):,} streaks  "
            f"valid={n_valid:,}  det={n_detectable:,}"
        )

        # Free band memory immediately
        del out, band_df, ab_mag, L_px, t_crossing_sec
        del sat_altitude_km, elevation_deg, ab_mag_band
        del theta_defocus, theta_eff, e_per_px, sb
        gc.collect()

    del df; gc.collect()

    elapsed = time.perf_counter() - t0
    return f"[DONE] {date_str}  {elapsed:.1f}s\n" + "\n".join(msgs)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(
    step5_dir  = "step8_output_pixel_loss",
    step2_dir  = "step2_output",
    step3_dir  = "step3_output",
    output_dir = "step6_output_perband",
    n_workers  = 36,   # days processed simultaneously
                       # each worker handles one full day sequentially
                       # 36 workers × 1 day = 36 cores used
):
    t_start = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] step5_dir  : {step5_dir}")
    print(f"[INFO] step2_dir  : {step2_dir}")
    print(f"[INFO] step3_dir  : {step3_dir}")
    print(f"[INFO] output_dir : {output_dir}")
    print(f"[INFO] n_workers  : {n_workers}")

    # Match step5 files to step2 files by date
    step5_paths = sorted(glob.glob(
        os.path.join(step5_dir, "step5_datapoints_*.csv")
    ))
    if not step5_paths:
        raise FileNotFoundError(f"No step5_datapoints_*.csv in {step5_dir}")

    step2_map = {
        os.path.basename(p)
        .replace("streak_trajectories_", "")
        .replace(".csv", ""): p
        for p in sorted(glob.glob(
            os.path.join(step2_dir, "streak_trajectories_*.csv")
        ))
        if "_azel" not in p
    }

    # Build task list — one task per day
    tasks = []
    for step5_path in step5_paths:
        date_str = (os.path.basename(step5_path)
                    .replace("step5_datapoints_", "")
                    .replace(".csv", ""))

        if date_str not in step2_map:
            print(f"  [WARN] No step2 for {date_str} — skipping")
            continue

        # Skip if all bands already done
        all_done = all(
            os.path.exists(os.path.join(output_dir, f"step6_{date_str}_{b}.csv"))
            for b in BAND_PARAMS
        )
        if all_done:
            continue

        tasks.append((date_str, step5_path, step2_map[date_str],
                      step3_dir, output_dir))

    print(f"[INFO] {len(tasks)} days to process ({313 - len(tasks)} already done)\n")

    if not tasks:
        print("[INFO] Nothing to do — all files already exist.")
        return

    # Run: one worker per day, processes all bands for that day sequentially
    n = min(n_workers, cpu_count(), len(tasks))
    print(f"[INFO] Running {n} parallel workers …\n")

    with Pool(processes=n) as pool:
        for msg in pool.imap_unordered(process_one_day, tasks):
            print(msg, flush=True)

    elapsed = time.perf_counter() - t_start
    all_out  = sorted(glob.glob(os.path.join(output_dir, "step6_*.csv")))
    total_gb = sum(os.path.getsize(p) for p in all_out) / 1e9

    print(f"\n{'='*60}")
    print(f"[ALL DONE]  {len(all_out)} files  {total_gb:.2f} GB  {elapsed:.1f}s")
    for band in "ugrizy":
        n_files = len(glob.glob(os.path.join(output_dir, f"step6_*_{band}.csv")))
        print(f"  {band}: {n_files} day files")
    print(f"{'='*60}")


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import pytz
from datetime import datetime

CHILE_TZ    = pytz.timezone("America/Santiago")
BIN_MINUTES = 2
OUTPUT_PNG  = "surface_brightness_heatmap.png"


def _bin_hour(hour, bin_minutes):
    return np.floor(hour / (bin_minutes / 60.0)) * (bin_minutes / 60.0)


def _utc_to_chile_hour(date, utc_hour):
    dt_utc = pytz.utc.localize(
        datetime(date.year, date.month, date.day) +
        pd.Timedelta(hours=float(utc_hour))
    )
    dt_chile = dt_utc.astimezone(CHILE_TZ)
    return dt_chile.hour + dt_chile.minute / 60.0


def _chile_to_display(h):
    return (h - 12.0) % 24.0


def collect_records_from_daywise(step6_output_dir, step2_dir, bin_minutes):
    print(f"\n[PLOT] Building t_mjd lookup from step2 …")
    step2_paths = [p for p in sorted(glob.glob(
        os.path.join(step2_dir, "streak_trajectories_*.csv")
    )) if "_azel" not in p]

    tmjd_rows = []
    for i, path in enumerate(step2_paths):
        date_str = (os.path.basename(path)
                    .replace("streak_trajectories_", "")
                    .replace(".csv", ""))
        try:
            df = pd.read_csv(path, usecols=[
                "pointing_id", "sat_name", "shell_id",
                "t_mjd", "in_fov", "sunlit"
            ])
        except Exception:
            continue

        mask = (df["in_fov"].astype(str).str.lower() == "true") & \
               (df["sunlit"].astype(str).str.lower() == "true")
        df = df[mask].copy()
        if df.empty:
            del df; gc.collect(); continue

        agg = (df.groupby(["pointing_id", "sat_name", "shell_id"])["t_mjd"]
                 .median().reset_index())
        agg["date_str"] = date_str
        tmjd_rows.append(agg)
        del df, agg; gc.collect()

        if (i + 1) % 50 == 0 or i == len(step2_paths) - 1:
            print(f"  {i+1}/{len(step2_paths)} step2 files scanned …")

    tmjd_df = pd.concat(tmjd_rows, ignore_index=True)
    print(f"  t_mjd lookup: {len(tmjd_df):,} streaks")
    del tmjd_rows; gc.collect()

    step6_files = sorted(glob.glob(os.path.join(step6_output_dir, "step6_*.csv")))
    print(f"[PLOT] Processing {len(step6_files)} step6 files …")

    records = []
    for i, fpath in enumerate(step6_files):
        try:
            chunk = pd.read_csv(fpath, usecols=[
                "pointing_id", "sat_name", "shell_id",
                "surface_brightness_mag_arcsec2"
            ])
        except Exception:
            continue

        chunk = chunk[chunk["surface_brightness_mag_arcsec2"].notna()].copy()
        if chunk.empty:
            continue

        chunk = chunk.merge(
            tmjd_df[["pointing_id", "sat_name", "shell_id", "t_mjd", "date_str"]],
            on=["pointing_id", "sat_name", "shell_id"], how="left"
        )
        chunk = chunk[chunk["t_mjd"].notna()].copy()
        if chunk.empty:
            continue

        chunk["date_ts"]        = pd.to_datetime(chunk["date_str"])
        chunk["hour_utc"]       = (chunk["t_mjd"] % 1.0) * 24.0
        chunk["hour_bin_utc"]   = chunk["hour_utc"].apply(lambda h: _bin_hour(h, bin_minutes))
        chunk["hour_bin_chile"] = chunk.apply(
            lambda row: _bin_hour(
                _utc_to_chile_hour(row["date_ts"], row["hour_bin_utc"]),
                bin_minutes
            ), axis=1
        )

        agg = (chunk.groupby(["date_str", "hour_bin_chile"])
                    ["surface_brightness_mag_arcsec2"]
                    .agg(["mean", "min"]).reset_index())
        agg.columns = ["date", "hour_bin", "avg_sb", "min_sb"]
        records.append(agg)
        del chunk, agg; gc.collect()

        if (i + 1) % 100 == 0 or i == len(step6_files) - 1:
            print(f"  {i+1}/{len(step6_files)} files processed …")

    result = pd.concat(records, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    result = (result.groupby(["date", "hour_bin"])
                    .agg(avg_sb=("avg_sb", "mean"), min_sb=("min_sb", "min"))
                    .reset_index())

    print(f"  Records: {len(result):,}  SB range: "
          f"{result['min_sb'].min():.2f} → {result['avg_sb'].max():.2f}")
    return result


def build_grid(df, bin_minutes):
    all_dates = sorted(df["date"].unique())
    bin_h     = bin_minutes / 60.0
    all_hours = np.arange(0.0, 24.0, bin_h)
    grid_avg    = np.full((len(all_hours), len(all_dates)), np.nan)
    grid_bright = np.full((len(all_hours), len(all_dates)), np.nan)
    date_idx = {d: i for i, d in enumerate(all_dates)}
    hour_idx = {round(h, 6): i for i, h in enumerate(all_hours)}
    for _, row in df.iterrows():
        d_i = date_idx.get(row["date"])
        h_i = hour_idx.get(round(row["hour_bin"], 6))
        if d_i is not None and h_i is not None:
            grid_avg   [h_i, d_i] = row["avg_sb"]
            grid_bright[h_i, d_i] = row["min_sb"]
    return np.array(all_dates, dtype="datetime64[D]"), all_hours, grid_avg, grid_bright


def plot_heatmap(dates, hours, grid_avg, grid_bright, output_png, bin_minutes):
    hours_display = np.array([_chile_to_display(h) for h in hours])
    sort_idx      = np.argsort(hours_display)
    h_disp        = hours_display[sort_idx]
    g_avg         = grid_avg[sort_idx, :]
    g_bright      = grid_bright[sort_idx, :]
    dates_mpl     = mdates.date2num(pd.to_datetime(dates))
    X, Y          = np.meshgrid(dates_mpl, h_disp)

    turbo_clipped = LinearSegmentedColormap.from_list(
        "turbo_clipped", cm.turbo_r(np.linspace(0.1, 0.95, 256))
    )

    plt.rcParams.update({"font.size": 20})
    fig, axes = plt.subplots(2, 1, figsize=(20, 14), sharex=True)
    fig.patch.set_facecolor("white")

    for grid, title, ax in [
        (g_avg,    "Mean Surface Brightness per 2-min Bin",      axes[0]),
        (g_bright, "Brightest Surface Brightness per 2-min Bin", axes[1]),
    ]:
        mask = ~np.isnan(grid)
        sc = ax.scatter(X[mask], Y[mask], c=grid[mask],
                        cmap=turbo_clipped, vmin=9.0, vmax=24.5,
                        s=20, linewidths=0, rasterized=True)
        cbar = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.02)
        cbar.set_label("Surface Brightness\n(mag/arcsec²)", fontsize=16)
        cbar.set_ticks(np.round(np.linspace(9.0, 24.5, 7), 1).tolist())
        cbar.ax.invert_yaxis()
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.set_ylabel("Chile local time (CLT/CLST)", fontsize=16)
        ax.set_facecolor("white")
        ax.set_ylim(6.0, 20.0)
        ticks = np.arange(6.0, 20.1, 2)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{int((t+12)%24):02d}:00" for t in ticks], fontsize=16)
        ax.axhline(12, color="black", lw=1.2, ls="--", alpha=0.5)
        ax.grid(axis="both", alpha=0.2, color="gray")

    axes[1].set_xlabel("Date", fontsize=18)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha="right")
    fig.text(0.99, 0.005,
             f"Time bin: {bin_minutes} min  |  Chile local time (CLT/CLST)",
             ha="right", va="bottom", fontsize=13, color="gray")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"\n[PLOT] Saved: {output_png}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    STEP5_DIR  = "step5_output_pixel_loss_rad"
    STEP2_DIR  = "step2_output"
    STEP3_DIR  = "step3_output"
    OUTPUT_DIR = "step6_output_perband_rad"

    # Step 1: compute brightness — one file at a time, 36 days in parallel
    main(
        step5_dir  = STEP5_DIR,
        step2_dir  = STEP2_DIR,
        step3_dir  = STEP3_DIR,
        output_dir = OUTPUT_DIR,
        n_workers  = 94,
    )

    # Step 2: collect + plot
    print("\n" + "="*60)
    print("  Collecting records for plot …")
    print("="*60)
    records_df = collect_records_from_daywise(OUTPUT_DIR, STEP2_DIR, BIN_MINUTES)

    print("\n" + "="*60)
    print("  Building grids …")
    print("="*60)
    dates, hours, grid_avg, grid_bright = build_grid(records_df, BIN_MINUTES)
    pct = 100 * np.sum(~np.isnan(grid_avg)) / grid_avg.size
    print(f"  Grid: {grid_avg.shape}  fill: {pct:.1f}%")

    print("\n" + "="*60)
    print("  Plotting …")
    print("="*60)
    plot_heatmap(dates, hours, grid_avg, grid_bright, OUTPUT_PNG, BIN_MINUTES)

    print("\n[ALL DONE]")