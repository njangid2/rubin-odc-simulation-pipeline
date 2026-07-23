"""
step5_pixel_loss.py  (v2 — reads surface brightness from step6 output)
─────────────────────────────────────────────────────────────────────────────
Computes satellite streak pixel loss for every pointing.

CHANGES FROM v1
───────────────
    v1 recomputed surface brightness from ab_magnitude using a simplified
    single-band formula with wrong throughput and no solar color correction.

    v2 reads surface_brightness_mag_arcsec2 directly from step6 output,
    which already has correct per-band throughput, solar color correction,
    slant-range defocus, and PSF. This avoids recomputing SB and removes
    the dependency on step4 bright files.

FIX (this version)
───────────────────
    pointing_filter in step2 is the raw opsim value (band letter + filter
    load/throughput-curve id, e.g. "r_57", "g_12") — NOT a bare single
    letter. The original v2 code used this raw value directly as the
    `band` key into BAND_ZP and sb_to_peak_electrons(), which only has
    plain single-letter keys ('u','g','r','i','z','y'). Since "r_57" is
    never a member of BAND_ZP, sb_to_peak_electrons() always returned NaN
    for EVERY streak regardless of actual brightness, so classify_status()
    always returned "faint" — saturated/unsaturated counts were always 0
    across the entire survey. Fixed by stripping the suffix
    (str.split("_")[0]) at the point pointing_filter is read, exactly as
    step5_pixel_loss.py v1's base_band() helper already does.

WORKFLOW PER STREAK
───────────────────
    1. Read sb from step6 output (already per-band corrected)
    2. If sb is NaN → faint → all pixel losses = 0
    3. If sb valid → compute peak_electrons from sb
    4. If peak_e > FULL_WELL_E → saturated, else unsaturated
    5. Compute pixel loss from L_px and mask width

INPUTS (per day)
────────────────
    step2_output/streak_trajectories_YYYY-MM-DD.csv       (geometry/L_px)
    step6_output_perband/step6_YYYY-MM-DD_{band}.csv      (surface brightness)

OUTPUTS (per day)
─────────────────
    step8_output_pixel_loss_v2/step5_datapoints_YYYY-MM-DD.csv
    step8_output_pixel_loss_v2/step5_pointings_YYYY-MM-DD.csv
    step8_output_pixel_loss_v2/step5_daily_summary.csv
"""

import os, gc, glob, time, logging, sys, math
from multiprocessing import Pool

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str, log_filename: str = "step5.log") -> str:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()
    fmt = logging.Formatter(
        fmt="%(asctime)s  [PID %(process)5d]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return log_path

log = logging.info

def _hms(s):
    h, r = divmod(int(s), 3600); m, sec = divmod(r, 60); frac = s - int(s)
    if h:  return f"{h}h {m}m {sec+frac:.2f}s"
    if m:  return f"{m}m {sec+frac:.2f}s"
    return f"{sec+frac:.2f}s"

# ─────────────────────────────────────────────────────────────────────────────
# Filter base-band helper
# ─────────────────────────────────────────────────────────────────────────────

def base_band(filter_value) -> str:
    """
    Extracts the base LSST band letter from a (possibly suffixed)
    pointing_filter value, e.g. "r_57" -> "r", "g_12" -> "g", "u" -> "u".
    """
    return str(filter_value).strip().split('_')[0]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PLATE_SCALE_ARCSEC_PX = 0.2
PSF_FWHM_ARCSEC       = 0.67
PSF_FWHM_PX           = PSF_FWHM_ARCSEC / PLATE_SCALE_ARCSEC_PX   # 3.35 px
PSF_PEAK_FRAC         = 0.44
PIXEL_AREA_ARCSEC2    = PLATE_SCALE_ARCSEC_PX ** 2
TOTAL_PIXELS          = 3.2e9
FOV_RADIUS_DEG        = 1.75
FULL_WELL_E           = 130_000

# Per-band zero points (SMTN-002) — needed to invert SB → electrons
BAND_ZP = {
    'u': 26.52, 'g': 28.51, 'r': 28.36,
    'i': 28.17, 'z': 27.78, 'y': 26.82,
}

MASK_WIDTH = {
    "saturated":   500,
    "unsaturated": 100,
    "faint":       0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Physics — invert SB to peak electrons
# ─────────────────────────────────────────────────────────────────────────────

def sb_to_peak_electrons(sb, band, t_crossing_sec):
    """
    Inverts surface brightness formula to get peak electrons per pixel.

    SB = ZP - 2.5 * log10(flux_per_arcsec2)
    → flux_per_arcsec2 = 10^((ZP - SB) / 2.5)   [e/s/arcsec²]
    → e_per_px_per_sec = flux_per_arcsec2 * pixel_area_arcsec2
    → e_per_px         = e_per_px_per_sec * t_crossing_sec
    → peak_e           = e_per_px * PSF_PEAK_FRAC

    PSF_PEAK_FRAC = 0.44 accounts for the fraction of total PSF flux
    concentrated in the peak pixel of a Gaussian PSF.
    """
    if (np.isnan(sb) or np.isnan(t_crossing_sec)
            or t_crossing_sec <= 0 or band not in BAND_ZP):
        return np.nan

    zp               = BAND_ZP[band]
    flux_per_arcsec2 = 10.0 ** ((zp - sb) / 2.5)   # e/s/arcsec²
    e_per_px_per_sec = flux_per_arcsec2 * PIXEL_AREA_ARCSEC2
    e_per_px         = e_per_px_per_sec * t_crossing_sec
    return e_per_px * PSF_PEAK_FRAC


def classify_status(sb, band, t_crossing_sec):
    """
    faint       → sb is NaN (step6 already excluded below detection limit)
    saturated   → peak electrons > FULL_WELL_E
    unsaturated → peak electrons <= FULL_WELL_E
    """
    if np.isnan(sb):
        return "faint", np.nan

    peak_e = sb_to_peak_electrons(sb, band, t_crossing_sec)
    if np.isnan(peak_e):
        return "faint", np.nan

    status = "saturated" if peak_e > FULL_WELL_E else "unsaturated"
    return status, peak_e

# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (unchanged from original step5)
# ─────────────────────────────────────────────────────────────────────────────

def angular_sep_deg(ra1, dec1, ra2, dec2):
    r1 = np.radians(ra1);  d1 = np.radians(dec1)
    r2 = np.radians(ra2);  d2 = np.radians(dec2)
    cos_angle = (np.sin(d1)*np.sin(d2) +
                 np.cos(d1)*np.cos(d2)*np.cos(r1-r2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def radec_to_focal_px(ra_deg, dec_deg, ra0_deg, dec0_deg):
    ra  = np.radians(ra_deg);  dec  = np.radians(dec_deg)
    ra0 = np.radians(ra0_deg); dec0 = np.radians(dec0_deg)
    cos_c = (np.sin(dec0)*np.sin(dec) +
             np.cos(dec0)*np.cos(dec)*np.cos(ra-ra0))
    x_rad = np.cos(dec)*np.sin(ra-ra0) / cos_c
    y_rad = (np.cos(dec0)*np.sin(dec) -
             np.sin(dec0)*np.cos(dec)*np.cos(ra-ra0)) / cos_c
    x_px  = np.degrees(x_rad) * 3600.0 / PLATE_SCALE_ARCSEC_PX
    y_px  = np.degrees(y_rad) * 3600.0 / PLATE_SCALE_ARCSEC_PX
    return x_px, y_px


def interpolate_fov_boundary(ra_in, dec_in, ra_out, dec_out,
                              pt_ra, pt_dec,
                              fov_radius_deg=FOV_RADIUS_DEG):
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid   = 0.5 * (lo + hi)
        ra_m  = ra_in  + mid*(ra_out  - ra_in)
        dec_m = dec_in + mid*(dec_out - dec_in)
        sep   = angular_sep_deg(pt_ra, pt_dec, ra_m, dec_m)
        if sep < fov_radius_deg: lo = mid
        else:                    hi = mid
    f = 0.5*(lo+hi)
    return ra_in + f*(ra_out-ra_in), dec_in + f*(dec_out-dec_in)


def streak_polygon(x0, y0, x1, y1, half_width_px):
    dx = x1-x0; dy = y1-y0
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    nx = -dy/length; ny = dx/length
    corners = [
        (x0+nx*half_width_px, y0+ny*half_width_px),
        (x1+nx*half_width_px, y1+ny*half_width_px),
        (x1-nx*half_width_px, y1-ny*half_width_px),
        (x0-nx*half_width_px, y0-ny*half_width_px),
    ]
    return Polygon(corners)

# ─────────────────────────────────────────────────────────────────────────────
# Process ONE streak group → File 1 row
# ─────────────────────────────────────────────────────────────────────────────

def process_streak(grp, pt_ra, pt_dec, filt, t_exp, sb_lookup):
    """
    grp        : step2 rows for this (pointing_id, sat_name, shell_id)
    sb_lookup  : dict keyed by (pointing_id, sat_name, shell_id) →
                 {"sb": float, "t_crossing_sec": float, "L_px": float}
    """
    grp = grp.sort_values("step").reset_index(drop=True)

    in_fov_mask = grp["in_fov"].astype(bool)
    in_fov_idx  = grp.index[in_fov_mask].tolist()
    if not in_fov_idx:
        return None

    key = (int(grp["pointing_id"].iloc[0]),
           grp["sat_name"].iloc[0],
           int(grp["shell_id"].iloc[0]))

    # ── Pull SB and geometry from step6 lookup ────────────────────────────
    s6 = sb_lookup.get(key, {})
    sb            = s6.get("sb",             np.nan)
    t_crossing_sec = s6.get("t_crossing_sec", np.nan)
    L_px          = s6.get("L_px",           np.nan)
    ab_mag        = s6.get("ab_magnitude",   np.nan)

    # ── Classify using step6 SB ───────────────────────────────────────────
    status, peak_e = classify_status(sb, filt, t_crossing_sec)

    # ── Pixel loss ────────────────────────────────────────────────────────
    if status == "faint" or np.isnan(L_px) or L_px <= 0:
        pl_psf         = 0.0
        pl_saturated   = 0.0
        pl_unsaturated = 0.0
        pl_recommended = 0.0
        W_mask         = 0
    else:
        pl_psf         = L_px * PSF_FWHM_PX
        pl_saturated   = L_px * MASK_WIDTH["saturated"]
        pl_unsaturated = L_px * MASK_WIDTH["unsaturated"]
        pl_recommended = (pl_saturated if status == "saturated"
                          else pl_unsaturated)
        W_mask         = MASK_WIDTH[status]

    # ── Focal-plane endpoints for Shapely union ────────────────────────────
    # Use actual in-FOV trajectory endpoints for geometry
    first_in = in_fov_idx[0];  last_in = in_fov_idx[-1]
    first_step = grp.index[0]; last_step = grp.index[-1]

    entering = (first_in > first_step)
    exiting  = (last_in  < last_step)

    if   entering and exiting: streak_type = "both"
    elif entering:             streak_type = "entering"
    elif exiting:              streak_type = "exiting"
    else:                      streak_type = "full"

    ra_in  = grp.loc[in_fov_idx, "ra_deg"].values
    dec_in = grp.loc[in_fov_idx, "dec_deg"].values

    ra_entry, dec_entry = ra_in[0], dec_in[0]
    ra_exit,  dec_exit  = ra_in[-1], dec_in[-1]

    if entering and first_in > first_step:
        ra_out  = float(grp.loc[first_in-1, "ra_deg"])
        dec_out = float(grp.loc[first_in-1, "dec_deg"])
        ra_entry, dec_entry = interpolate_fov_boundary(
            ra_in[0], dec_in[0], ra_out, dec_out, pt_ra, pt_dec)

    if exiting and last_in < last_step:
        ra_out  = float(grp.loc[last_in+1, "ra_deg"])
        dec_out = float(grp.loc[last_in+1, "dec_deg"])
        ra_exit, dec_exit = interpolate_fov_boundary(
            ra_in[-1], dec_in[-1], ra_out, dec_out, pt_ra, pt_dec)

    x0, y0 = radec_to_focal_px(ra_entry, dec_entry, pt_ra, pt_dec)
    x1, y1 = radec_to_focal_px(ra_exit,  dec_exit,  pt_ra, pt_dec)

    return {
        "pointing_id":                    int(grp["pointing_id"].iloc[0]),
        "sat_name":                       grp["sat_name"].iloc[0],
        "shell_id":                       int(grp["shell_id"].iloc[0]),
        "pointing_filter":                filt,
        "pointing_exptime":               t_exp,
        "ab_magnitude":                   round(ab_mag, 4) if not np.isnan(ab_mag) else np.nan,
        "streak_type":                    streak_type,
        "L_px":                           round(L_px, 2) if not np.isnan(L_px) else np.nan,
        "t_crossing_sec":                 round(t_crossing_sec, 4) if not np.isnan(t_crossing_sec) else np.nan,
        "surface_brightness_mag_arcsec2": round(sb, 4) if not np.isnan(sb) else np.nan,
        "peak_electrons":                 round(peak_e, 1) if not np.isnan(peak_e) else np.nan,
        "sat_status":                     status,
        "pixel_loss_psf":                 round(pl_psf, 2),
        "pixel_loss_saturated":           round(pl_saturated, 2),
        "pixel_loss_unsaturated":         round(pl_unsaturated, 2),
        "pixel_loss_recommended":         round(pl_recommended, 2),
        # internal for Shapely
        "_x0": x0, "_y0": y0, "_x1": x1, "_y1": y1,
        "_W_mask": W_mask,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Process ONE pointing → File 2 row
# ─────────────────────────────────────────────────────────────────────────────

def process_pointing(pid, streak_rows, pt_meta):
    statuses      = [r["sat_status"] for r in streak_rows]
    n_saturated   = statuses.count("saturated")
    n_unsaturated = statuses.count("unsaturated")
    n_faint       = statuses.count("faint")

    psf_polys = []
    rec_polys = []

    for r in streak_rows:
        if r["sat_status"] == "faint":
            continue
        x0, y0 = r["_x0"], r["_y0"]
        x1, y1 = r["_x1"], r["_y1"]
        W       = r["_W_mask"]

        poly_psf = streak_polygon(x0, y0, x1, y1, PSF_FWHM_PX / 2.0)
        poly_rec = streak_polygon(x0, y0, x1, y1, W / 2.0)

        if poly_psf and poly_psf.is_valid:
            psf_polys.append(poly_psf)
        if poly_rec and poly_rec.is_valid:
            rec_polys.append(poly_rec)

    psf_union_area = unary_union(psf_polys).area if psf_polys else 0.0
    rec_union_area = unary_union(rec_polys).area if rec_polys else 0.0

    psf_fraction = min(psf_union_area / TOTAL_PIXELS, 1.0)
    rec_fraction = min(rec_union_area / TOTAL_PIXELS, 1.0)

    return {
        "pointing_id":                           pid,
        "pointing_ra":                           round(pt_meta["pointing_ra"], 5),
        "pointing_dec":                          round(pt_meta["pointing_dec"], 5),
        "pointing_filter":                       pt_meta["pointing_filter"],
        "pointing_exptime":                      pt_meta["pointing_exptime"],
        "n_streaks":                             len(streak_rows),
        "total_pixel_loss_psf_fraction":         round(psf_fraction, 8),
        "total_pixel_loss_recommended_fraction": round(rec_fraction, 8),
        "n_saturated":                           n_saturated,
        "n_unsaturated":                         n_unsaturated,
        "n_faint":                               n_faint,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Column definitions
# ─────────────────────────────────────────────────────────────────────────────

FILE1_COLS = [
    "pointing_id", "sat_name", "shell_id", "pointing_filter", "pointing_exptime",
    "ab_magnitude", "streak_type", "L_px", "t_crossing_sec",
    "surface_brightness_mag_arcsec2", "peak_electrons",
    "sat_status",
    "pixel_loss_psf", "pixel_loss_saturated",
    "pixel_loss_unsaturated", "pixel_loss_recommended",
]

FILE2_COLS = [
    "pointing_id", "pointing_ra", "pointing_dec", "pointing_filter",
    "pointing_exptime", "n_streaks",
    "total_pixel_loss_psf_fraction",
    "total_pixel_loss_recommended_fraction",
    "n_saturated", "n_unsaturated", "n_faint",
]

FILE3_COLS = [
    "date", "pointing_night", "n_pointings", "n_affected_pointings",
    "mean_pixel_loss_psf_fraction", "max_pixel_loss_psf_fraction",
    "mean_pixel_loss_recommended_fraction", "max_pixel_loss_recommended_fraction",
    "total_saturated", "total_unsaturated", "total_faint",
]

# ─────────────────────────────────────────────────────────────────────────────
# Worker — processes ONE day
# ─────────────────────────────────────────────────────────────────────────────

def process_day(args):
    (step2_path, step6_dir, output_dir, log_path) = args
    setup_logging(output_dir, os.path.basename(log_path))

    date_str = (os.path.basename(step2_path)
                .replace("streak_trajectories_", "").replace(".csv", ""))

    f1_path = os.path.join(output_dir, f"step5_datapoints_{date_str}.csv")
    f2_path = os.path.join(output_dir, f"step5_pointings_{date_str}.csv")

    t0 = time.perf_counter()
    log(f"[START] {date_str}")

    if os.path.isfile(f1_path) and os.path.isfile(f2_path):
        log(f"[SKIP]  {date_str} — output files exist")
        return {"date": date_str, "status": "skipped"}

    # ── Load step2 ────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(step2_path)
    except Exception as e:
        log(f"[ERROR] {date_str} — step2 read failed: {e}")
        return {"date": date_str, "status": "error"}

    N = len(df)
    if N == 0:
        pd.DataFrame(columns=FILE1_COLS).to_csv(f1_path, index=False)
        pd.DataFrame(columns=FILE2_COLS).to_csv(f2_path, index=False)
        return {"date": date_str, "status": "done_empty"}

    log(f"  {date_str} | {N:,} step2 rows loaded")

    # ── Load step6 band files for this day → build SB lookup ─────────────
    # Lookup key: (pointing_id, sat_name, shell_id)
    # Value: {sb, t_crossing_sec, L_px, ab_magnitude}
    sb_lookup = {}
    bands_found = []

    for band in "ugrizy":
        s6_path = os.path.join(step6_dir, f"step6_{date_str}_{band}.csv")
        if not os.path.exists(s6_path):
            continue
        try:
            s6 = pd.read_csv(s6_path, usecols=[
                "pointing_id", "sat_name", "shell_id",
                "surface_brightness_mag_arcsec2",
                "t_crossing_sec", "L_px", "ab_magnitude"
            ])
            bands_found.append(band)
        except Exception as e:
            log(f"  [WARN] {date_str} {band}: step6 read failed: {e}")
            continue

        for _, row in s6.iterrows():
            key = (int(row["pointing_id"]),
                   row["sat_name"],
                   int(row["shell_id"]))
            sb_lookup[key] = {
                "sb":             row["surface_brightness_mag_arcsec2"],
                "t_crossing_sec": row["t_crossing_sec"],
                "L_px":           row["L_px"],
                "ab_magnitude":   row["ab_magnitude"],
            }
        del s6; gc.collect()

    log(f"  {date_str} | step6 bands loaded: {bands_found}  "
        f"SB entries: {len(sb_lookup):,}")

    # ── Process streaks ───────────────────────────────────────────────────
    grouped  = df.groupby(["pointing_id", "sat_name", "shell_id"])
    n_groups = len(grouped)
    log(f"  {date_str} | {n_groups:,} streak groups")

    file1_rows   = []
    pointing_map = {}

    for i, ((pid, sat, shell), grp) in enumerate(grouped):
        pt_row = grp.iloc[0]
        pt_ra  = float(pt_row["pointing_ra"])
        pt_dec = float(pt_row["pointing_dec"])
        filt   = base_band(pt_row["pointing_filter"])   # FIX: strip "_NN" suffix, e.g. "r_57" -> "r"
        t_exp  = float(pt_row["pointing_exptime"])

        result = process_streak(grp, pt_ra, pt_dec, filt, t_exp, sb_lookup)
        if result is None:
            continue

        f1_row = {k: v for k, v in result.items() if not k.startswith("_")}
        file1_rows.append(f1_row)

        if pid not in pointing_map:
            pointing_map[pid] = {
                "meta": {
                    "pointing_ra":      pt_ra,
                    "pointing_dec":     pt_dec,
                    "pointing_filter":  filt,
                    "pointing_exptime": t_exp,
                },
                "streaks": [],
            }
        pointing_map[pid]["streaks"].append(result)

        if (i + 1) % 500 == 0 or i == n_groups - 1:
            log(f"  {date_str} | streaks {i+1:,}/{n_groups:,}")

    # ── Save File 1 ───────────────────────────────────────────────────────
    f1_df = pd.DataFrame(file1_rows, columns=FILE1_COLS)
    f1_df.to_csv(f1_path, index=False)
    log(f"  {date_str} | File1 saved: {len(f1_df):,} streaks")

    # ── Build and save File 2 ─────────────────────────────────────────────
    file2_rows = []
    n_pts = len(pointing_map)
    log(f"  {date_str} | building Shapely unions for {n_pts:,} pointings …")

    for j, (pid, pdata) in enumerate(pointing_map.items()):
        file2_rows.append(
            process_pointing(pid, pdata["streaks"], pdata["meta"]))
        if (j + 1) % 200 == 0 or j == n_pts - 1:
            log(f"  {date_str} | pointings {j+1:,}/{n_pts:,}")

    f2_df = pd.DataFrame(file2_rows, columns=FILE2_COLS)
    f2_df.to_csv(f2_path, index=False)
    log(f"  {date_str} | File2 saved: {len(f2_df):,} pointings")

    elapsed = time.perf_counter() - t0
    sc = f1_df["sat_status"].value_counts().to_dict()
    log(f"[DONE]  {date_str} | {_hms(elapsed)}  "
        f"sat={sc.get('saturated',0):,}  "
        f"unsat={sc.get('unsaturated',0):,}  "
        f"faint={sc.get('faint',0):,}")

    del df, f1_df, f2_df, sb_lookup
    gc.collect()

    return {"date": date_str, "status": "done",
            "n_streaks": len(file1_rows), "n_pointings": n_pts,
            "elapsed": elapsed}

# ─────────────────────────────────────────────────────────────────────────────
# Daily summary (File 3)
# ─────────────────────────────────────────────────────────────────────────────

def append_daily_summary(output_dir, date_str, step2_path):
    f2_path = os.path.join(output_dir, f"step5_pointings_{date_str}.csv")
    f3_path = os.path.join(output_dir, "step5_daily_summary.csv")

    if not os.path.isfile(f2_path):
        return
    try:
        f2 = pd.read_csv(f2_path)
    except pd.errors.EmptyDataError:
        return
    if f2.empty:
        return

    try:
        night = int(pd.read_csv(step2_path, usecols=["pointing_night"],
                                nrows=1)["pointing_night"].iloc[0])
    except Exception:
        night = -1

    row = {
        "date":                                date_str,
        "pointing_night":                      night,
        "n_pointings":                         len(f2),
        "n_affected_pointings":                int((f2["n_streaks"] > 0).sum()),
        "mean_pixel_loss_psf_fraction":
            round(f2["total_pixel_loss_psf_fraction"].mean(), 8),
        "max_pixel_loss_psf_fraction":
            round(f2["total_pixel_loss_psf_fraction"].max(), 8),
        "mean_pixel_loss_recommended_fraction":
            round(f2["total_pixel_loss_recommended_fraction"].mean(), 8),
        "max_pixel_loss_recommended_fraction":
            round(f2["total_pixel_loss_recommended_fraction"].max(), 8),
        "total_saturated":   int(f2["n_saturated"].sum()),
        "total_unsaturated": int(f2["n_unsaturated"].sum()),
        "total_faint":       int(f2["n_faint"].sum()),
    }

    write_header = not os.path.isfile(f3_path)
    pd.DataFrame([row], columns=FILE3_COLS).to_csv(
        f3_path, mode="a", header=write_header, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(
    step2_dir  = "step2_output",
    step6_dir  = "step6_output_perband",
    output_dir = "step5_output_v2",
    n_workers  = 40,
    log_filename = "step5.log",
):
    log_path   = setup_logging(output_dir, log_filename)
    wall_start = time.perf_counter()

    log("=" * 70)
    log("  STEP 5 v2: Pixel Loss from step6 Surface Brightness")
    log("=" * 70)
    log(f"  step2_dir  : {step2_dir}")
    log(f"  step6_dir  : {step6_dir}")
    log(f"  output_dir : {output_dir}")
    log(f"  n_workers  : {n_workers}")
    os.makedirs(output_dir, exist_ok=True)

    all_step2 = sorted(glob.glob(
        os.path.join(step2_dir, "streak_trajectories_*.csv")))
    all_step2 = [p for p in all_step2
                 if "_azel" not in p and "_bright" not in p]

    if not all_step2:
        log(f"[WARN] No step2 CSVs found in {step2_dir}")
        return

    log(f"\n[INFO] {len(all_step2)} day CSV(s) found")

    # ── Sanity check: confirm base_band() actually produces known bands ────
    sample_path = all_step2[0]
    try:
        sample_filters = pd.read_csv(
            sample_path, usecols=["pointing_filter"], nrows=2000
        )["pointing_filter"].unique()
        sample_bands = sorted(set(base_band(f) for f in sample_filters))
        log(f"[INFO] Sample pointing_filter raw values: {list(sample_filters[:10])}")
        log(f"[INFO] Extracted base bands: {sample_bands}")
    except Exception as e:
        log(f"[WARN] Could not sanity-check filter values: {e}")

    # Build task list — skip days with both output files already done
    tasks = []
    n_skip = 0
    for step2_path in all_step2:
        date_str = (os.path.basename(step2_path)
                    .replace("streak_trajectories_", "").replace(".csv", ""))
        f1 = os.path.join(output_dir, f"step5_datapoints_{date_str}.csv")
        f2 = os.path.join(output_dir, f"step5_pointings_{date_str}.csv")
        if os.path.isfile(f1) and os.path.isfile(f2):
            n_skip += 1
            continue
        tasks.append((step2_path, step6_dir, output_dir, log_path))

    log(f"[INFO] {len(tasks)} days to process  ({n_skip} already done)\n")

    if not tasks:
        log("[INFO] Nothing to do.")
    else:
        n = min(n_workers, len(tasks))
        with Pool(processes=n) as pool:
            results = pool.map(process_day, tasks)

        log("\n[INFO] Building daily summary (File 3) …")
        for step2_path, *_ in tasks:
            date_str = (os.path.basename(step2_path)
                        .replace("streak_trajectories_", "").replace(".csv", ""))
            append_daily_summary(output_dir, date_str, step2_path)

        log("\n" + "=" * 70)
        n_done   = sum(1 for r in results if r["status"] == "done")
        n_errors = sum(1 for r in results if r["status"] == "error")
        log(f"  Done    : {n_done}")
        log(f"  Errors  : {n_errors}")
        if n_done > 0:
            done = [r for r in results if r["status"] == "done"]
            log(f"  Avg time/day : "
                f"{_hms(sum(r['elapsed'] for r in done)/n_done)}")

    # Final summary from File 3
    f3_path = os.path.join(output_dir, "step5_daily_summary.csv")
    if os.path.isfile(f3_path):
        f3 = pd.read_csv(f3_path)
        log(f"\n  Daily summary ({len(f3)} days):")
        log(f"  Mean pixel loss PSF         : "
            f"{f3['mean_pixel_loss_psf_fraction'].mean():.6f}")
        log(f"  Mean pixel loss recommended : "
            f"{f3['mean_pixel_loss_recommended_fraction'].mean():.6f}")
        log(f"  Max  pixel loss recommended : "
            f"{f3['max_pixel_loss_recommended_fraction'].max():.6f}")
        log(f"  Total saturated  : {f3['total_saturated'].sum():,}")
        log(f"  Total unsaturated: {f3['total_unsaturated'].sum():,}")
        log(f"  Total faint      : {f3['total_faint'].sum():,}")

    log(f"\n  Total wall time: {_hms(time.perf_counter() - wall_start)}")
    log("[DONE]")


if __name__ == "__main__":
    main(
        step2_dir  = "step2_output",
        step6_dir  = "step6_output_perband_rad",
        output_dir = "step5_output_v2_rad",
        n_workers  = 94,
    )