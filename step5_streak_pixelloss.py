"""
step5_pixel_loss.py
─────────────────────────────────────────────────────────────────────────────
Computes satellite streak brightness impact and pixel loss for every
pointing in each step2 day CSV.

INPUTS (per day)
────────────────
    step2_output/streak_trajectories_YYYY-MM-DD.csv        ← positions
    step3_output/streak_trajectories_YYYY-MM-DD_azel.csv   ← az/el (row-aligned)
    step4_output_20deg/streak_trajectories_YYYY-MM-DD_bright.csv ← ab_mag (row-aligned)

OUTPUTS (per day)
─────────────────
    step5_output/step5_datapoints_YYYY-MM-DD.csv    ← File 1: per streak
    step5_output/step5_pointings_YYYY-MM-DD.csv     ← File 2: per pointing
    step5_output/step5_daily_summary.csv            ← File 3: per day (appended)

FILE 1 columns (one row per unique streak = pointing × sat × shell):
    pointing_id, sat_name, shell_id, pointing_filter, pointing_filter_raw,
    pointing_exptime,
    ab_magnitude, streak_type, ang_vel_arcsec_s, L_px, peak_electrons,
    sat_status,
    pixel_loss_psf,
    pixel_loss_blooming, pixel_loss_saturated,
    pixel_loss_unsaturated, pixel_loss_faint,
    pixel_loss_recommended   ← copy of whichever sat_status applies

FILE 2 columns (one row per pointing):
    pointing_id, pointing_ra, pointing_dec, pointing_filter,
    pointing_filter_raw, pointing_exptime,
    n_streaks,
    total_pixel_loss_psf_fraction,
    total_pixel_loss_recommended_fraction,
    n_blooming, n_saturated, n_unsaturated, n_faint

FILE 3 columns (one row per day, appended):
    date, pointing_night, n_pointings, n_affected_pointings,
    mean_pixel_loss_psf_fraction, max_pixel_loss_psf_fraction,
    mean_pixel_loss_recommended_fraction, max_pixel_loss_recommended_fraction,
    total_blooming, total_saturated, total_unsaturated, total_faint

NOTE ON pointing_filter
────────────────────────
    The raw opsim `filter` column is suffixed (e.g. "r_57", "g_12" — band
    letter + filter-load/throughput-curve id), not a bare single letter.
    Every downstream consumer (step4b solar-color lookup, step6 BAND_PARAMS
    lookup) only ever cares about the base band letter ("u"/"g"/"r"/"i"/
    "z"/"y"). To avoid every future script having to re-implement the same
    "split on _ and take [0]" fix, step5 does that conversion ONCE, here,
    at read time:

        pointing_filter      -> base band letter only, e.g. "r"
        pointing_filter_raw  -> original suffixed value, e.g. "r_57"
                                 (kept for traceability only)

    Any script reading step5's output can treat pointing_filter as a plain
    single-letter band with no further processing needed.

USAGE
─────
    python step5_pixel_loss.py

Requirements:
    pip install shapely lumos
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
# Rubin constants
# ─────────────────────────────────────────────────────────────────────────────

PLATE_SCALE_ARCSEC_PX = 0.2          # arcsec per pixel
PSF_FWHM_ARCSEC       = 0.67         # median atmospheric PSF
PSF_FWHM_PX           = PSF_FWHM_ARCSEC / PLATE_SCALE_ARCSEC_PX   # 3.35 px
PSF_PEAK_FRAC         = 0.44         # fraction of flux in peak pixel (Gaussian)
TOTAL_PIXELS          = 3.2e9        # 3.2 Gpixels
FOV_RADIUS_DEG        = 1.75         # Rubin FOV radius
A_MIRROR_M2           = math.pi * (6.49 / 2) ** 2   # 33.1 m²
THROUGHPUT            = 0.4          # combined QE × optics
FULL_WELL_E           = 130_000       # electrons per pixel
                                     # SLAC/eotest measurement (conservative lower bound)
                                     # RTN-056 reports 130,000 e- via DM stack method;
                                     # SLAC reports 90,000 e- via eotest — we use the
                                     # conservative value so saturation is not underestimated.
H_PLANCK              = 6.626e-34    # J*s

# lumos constants — must match lumos.constants exactly so AB mag -> intensity
# round-trip is lossless (same values lumos used when computing ab_magnitude)
import lumos.constants as _lumos_const
_LUMOS_C      = _lumos_const.SPEED_OF_LIGHT   # m/s
_LUMOS_LAM    = _lumos_const.WAVELENGTH        # m  (single effective wavelength lumos uses)

# Mask widths (pixels) per saturation status
# Source: lsst/meas_algorithms maskStreaks.py (MaskStreaksTask)
#   saturated   -> saturatedDetectionsDilation = 250 px  (SAT+DETECTED dilation)
#   unsaturated -> 2 x nSigmaMask x sigma = 2 x 5 x 10  = 100 px  (Moffat profile default)
MASK_WIDTH = {
    "saturated":   500,
    "unsaturated": 100,
}

# ─────────────────────────────────────────────────────────────────────────────
# Angular separation (vectorised, degrees)
# ─────────────────────────────────────────────────────────────────────────────

def angular_sep_deg(ra1, dec1, ra2, dec2):
    """Angular separation in degrees. Inputs can be scalars or arrays."""
    r1 = np.radians(ra1);  d1 = np.radians(dec1)
    r2 = np.radians(ra2);  d2 = np.radians(dec2)
    cos_angle = (np.sin(d1) * np.sin(d2) +
                 np.cos(d1) * np.cos(d2) * np.cos(r1 - r2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

# ─────────────────────────────────────────────────────────────────────────────
# Gnomonic projection: (RA, Dec) → focal-plane (x_px, y_px)
# ─────────────────────────────────────────────────────────────────────────────

def radec_to_focal_px(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """
    Gnomonic (tangent-plane) projection.
    Returns (x_px, y_px) relative to focal plane centre.
    Positive x = East, positive y = North.
    """
    ra  = np.radians(ra_deg);  dec  = np.radians(dec_deg)
    ra0 = np.radians(ra0_deg); dec0 = np.radians(dec0_deg)
    cos_c = np.sin(dec0)*np.sin(dec) + np.cos(dec0)*np.cos(dec)*np.cos(ra - ra0)
    x_rad = np.cos(dec) * np.sin(ra - ra0) / cos_c
    y_rad = (np.cos(dec0)*np.sin(dec) - np.sin(dec0)*np.cos(dec)*np.cos(ra - ra0)) / cos_c
    x_px  = np.degrees(x_rad) * 3600.0 / PLATE_SCALE_ARCSEC_PX
    y_px  = np.degrees(y_rad) * 3600.0 / PLATE_SCALE_ARCSEC_PX
    return x_px, y_px

# ─────────────────────────────────────────────────────────────────────────────
# FOV boundary crossing (linear interpolation)
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_fov_boundary(ra_in, dec_in, ra_out, dec_out,
                              pt_ra, pt_dec, fov_radius_deg=FOV_RADIUS_DEG):
    """
    Binary-search along the segment (ra_in→ra_out) to find where it crosses
    the FOV boundary circle. Returns (ra_cross, dec_cross).
    ra_in/dec_in  = point INSIDE  FOV
    ra_out/dec_out = point OUTSIDE FOV
    """
    lo, hi = 0.0, 1.0
    for _ in range(30):   # 30 iterations → sub-arcsec accuracy
        mid = 0.5 * (lo + hi)
        ra_m  = ra_in  + mid * (ra_out  - ra_in)
        dec_m = dec_in + mid * (dec_out - dec_in)
        sep   = angular_sep_deg(pt_ra, pt_dec, ra_m, dec_m)
        if sep < fov_radius_deg:
            lo = mid
        else:
            hi = mid
    f = 0.5 * (lo + hi)
    return ra_in + f*(ra_out - ra_in), dec_in + f*(dec_out - dec_in)

# ─────────────────────────────────────────────────────────────────────────────
# Saturation from first principles
# ─────────────────────────────────────────────────────────────────────────────

def ab_mag_to_intensity(ab_mag: float) -> float:
    """
    Exact inverse of lumos.conversions.intensity_to_ab_mag.
    Returns intensity in W/m² using the same constants lumos used,
    so the round-trip ab_mag → intensity is lossless.

    lumos forward:  log_val   = intensity * λ / (c * 3631e-26)
                    ab_mag    = -2.5 * log10(log_val)
    Inverse:        log_val   = 10 ** (-ab_mag / 2.5)
                    intensity = log_val * c * 3631e-26 / λ
    """
    log_val   = 10.0 ** (-ab_mag / 2.5)
    intensity = log_val * _LUMOS_C * 3631e-26 / _LUMOS_LAM   # W/m²
    return intensity


def compute_peak_electrons(ab_mag: float, L_px: float, t_exp: float) -> float:
    """
    Returns peak electrons per pixel in the streak's central pixel.

    Uses lumos-consistent intensity inversion — no filter params needed
    because lumos already folded the spectral response into ab_magnitude.

    Steps:
        1. ab_mag → intensity (W/m²)        exact lumos inverse
        2. intensity → photons/s/m²         using lumos wavelength constant
        3. × mirror area × throughput × t_exp  → total photons collected
        4. ÷ L_px                           → photons per pixel along streak
        5. × PSF_PEAK_FRAC                  → peak pixel electrons
    """
    if np.isnan(ab_mag) or L_px <= 0 or t_exp <= 0:
        return np.nan

    intensity   = ab_mag_to_intensity(ab_mag)          # W/m²
    E_photon    = H_PLANCK * _LUMOS_C / _LUMOS_LAM     # J per photon
    phot_flux   = intensity / E_photon                 # photons/s/m²
    N_photons   = phot_flux * A_MIRROR_M2 * THROUGHPUT * t_exp
    peak_e      = (N_photons / L_px) * PSF_PEAK_FRAC
    return peak_e

def classify_status(peak_e):
    """Returns sat_status string: 'saturated' or 'unsaturated'."""
    if np.isnan(peak_e):
        return "unsaturated"
    if peak_e > FULL_WELL_E:
        return "saturated"
    return "unsaturated"

# ─────────────────────────────────────────────────────────────────────────────
# Build Shapely polygon for one streak (rotated rectangle in focal-plane px)
# ─────────────────────────────────────────────────────────────────────────────

def streak_polygon(x0, y0, x1, y1, half_width_px):
    """
    Returns a Shapely Polygon for the streak rectangle.
    (x0,y0) → (x1,y1) is the streak centreline in focal-plane pixels.
    half_width_px is half the mask width.
    """
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    # Unit perpendicular
    nx = -dy / length
    ny =  dx / length
    corners = [
        (x0 + nx * half_width_px, y0 + ny * half_width_px),
        (x1 + nx * half_width_px, y1 + ny * half_width_px),
        (x1 - nx * half_width_px, y1 - ny * half_width_px),
        (x0 - nx * half_width_px, y0 - ny * half_width_px),
    ]
    return Polygon(corners)

# ─────────────────────────────────────────────────────────────────────────────
# Process ONE streak group → returns dict for File 1
# ─────────────────────────────────────────────────────────────────────────────

def process_streak(grp, pt_ra, pt_dec, filt, filt_raw, t_exp):
    """
    grp : DataFrame for one (pointing_id, sat_name, shell_id), sorted by step.
    filt     : base band letter, e.g. "r"
    filt_raw : original suffixed value, e.g. "r_57" (kept for traceability)
    Returns a dict of File 1 columns, or None if streak has no in-FOV points.
    """
    grp = grp.sort_values("step").reset_index(drop=True)

    in_fov_mask  = grp["in_fov"].astype(bool)
    in_fov_idx   = grp.index[in_fov_mask].tolist()

    if not in_fov_idx:
        return None

    ab_mag = float(grp.loc[in_fov_idx, "ab_magnitude"].dropna().median()
                   if "ab_magnitude" in grp.columns else np.nan)

    first_in   = in_fov_idx[0]
    last_in    = in_fov_idx[-1]
    first_step = grp.index[0]
    last_step  = grp.index[-1]

    # ── Classify crossing type ─────────────────────────────────────────────
    entering = (first_in > first_step)   # there is a step outside before first_in
    exiting  = (last_in  < last_step)    # there is a step outside after last_in

    if entering and exiting:
        streak_type = "both"
    elif entering:
        streak_type = "entering"
    elif exiting:
        streak_type = "exiting"
    else:
        streak_type = "full"

    # ── Collect in-FOV positions ───────────────────────────────────────────
    ra_in  = grp.loc[in_fov_idx, "ra_deg"].values
    dec_in = grp.loc[in_fov_idx, "dec_deg"].values
    t_in   = grp.loc[in_fov_idx, "t_mjd"].values

    # ── Angular velocity (arcsec/s) ────────────────────────────────────────
    if len(in_fov_idx) >= 2:
        dtheta = angular_sep_deg(ra_in[:-1], dec_in[:-1],
                                 ra_in[1:],  dec_in[1:]) * 3600.0   # arcsec
        dt     = np.diff(t_in) * 86400.0                             # seconds
        valid  = dt > 0
        ang_vel = float(np.mean(dtheta[valid] / dt[valid])) if valid.any() else np.nan
    else:
        ang_vel = np.nan

    # ── In-FOV segment length ──────────────────────────────────────────────
    if len(in_fov_idx) >= 2:
        segs      = angular_sep_deg(ra_in[:-1], dec_in[:-1],
                                    ra_in[1:],  dec_in[1:]) * 3600.0  # arcsec
        L_infov   = float(segs.sum())
    else:
        L_infov = 0.0

    # ── Partial extensions ─────────────────────────────────────────────────
    partial_before_arcsec = 0.0
    ra_entry, dec_entry   = ra_in[0], dec_in[0]   # default: first in-FOV point

    if entering and first_in > first_step:
        prev_idx = first_in - 1
        ra_out   = float(grp.loc[prev_idx, "ra_deg"])
        dec_out  = float(grp.loc[prev_idx, "dec_deg"])
        ra_bnd, dec_bnd = interpolate_fov_boundary(
            ra_in[0], dec_in[0], ra_out, dec_out, pt_ra, pt_dec
        )
        partial_before_arcsec = angular_sep_deg(
            pt_ra, pt_dec,   # dummy — we want seg length
            ra_bnd, dec_bnd  # ← override below
        )
        # Correct: length of partial segment
        partial_before_arcsec = angular_sep_deg(
            ra_bnd, dec_bnd, ra_in[0], dec_in[0]
        ) * 3600.0
        ra_entry, dec_entry = ra_bnd, dec_bnd

    partial_after_arcsec = 0.0
    ra_exit, dec_exit    = ra_in[-1], dec_in[-1]   # default: last in-FOV point

    if exiting and last_in < last_step:
        next_idx = last_in + 1
        ra_out   = float(grp.loc[next_idx, "ra_deg"])
        dec_out  = float(grp.loc[next_idx, "dec_deg"])
        ra_bnd, dec_bnd = interpolate_fov_boundary(
            ra_in[-1], dec_in[-1], ra_out, dec_out, pt_ra, pt_dec
        )
        partial_after_arcsec = angular_sep_deg(
            ra_in[-1], dec_in[-1], ra_bnd, dec_bnd
        ) * 3600.0
        ra_exit, dec_exit = ra_bnd, dec_bnd

    # ── Total streak length ────────────────────────────────────────────────
    L_total_arcsec = partial_before_arcsec + L_infov + partial_after_arcsec
    L_px           = L_total_arcsec / PLATE_SCALE_ARCSEC_PX

    # ── Saturation from first principles ──────────────────────────────────
    peak_e  = compute_peak_electrons(ab_mag, max(L_px, 1.0), t_exp)
    status  = classify_status(peak_e)
    W_mask  = MASK_WIDTH[status]

    # ── Pixel loss values (both always computed) ───────────────────────────
    pl_psf         = L_px * PSF_FWHM_PX
    pl_saturated   = L_px * MASK_WIDTH["saturated"]
    pl_unsaturated = L_px * MASK_WIDTH["unsaturated"]

    pl_recommended = {
        "saturated":   pl_saturated,
        "unsaturated": pl_unsaturated,
    }[status]

    # ── Streak endpoints in focal-plane pixels ─────────────────────────────
    x0, y0 = radec_to_focal_px(ra_entry, dec_entry, pt_ra, pt_dec)
    x1, y1 = radec_to_focal_px(ra_exit,  dec_exit,  pt_ra, pt_dec)

    return {
        # identifiers
        "pointing_id":             int(grp["pointing_id"].iloc[0]),
        "sat_name":                grp["sat_name"].iloc[0],
        "shell_id":                int(grp["shell_id"].iloc[0]),
        "pointing_filter":         filt,       # base band letter only, e.g. "r"
        "pointing_filter_raw":     filt_raw,   # original e.g. "r_57", for reference
        "pointing_exptime":        t_exp,
        # brightness
        "ab_magnitude":            round(ab_mag, 4),
        # geometry
        "streak_type":             streak_type,
        "ang_vel_arcsec_s":        round(ang_vel, 4) if not np.isnan(ang_vel) else np.nan,
        "L_px":                    round(L_px, 2),
        "partial_before_arcsec":   round(partial_before_arcsec, 3),
        "partial_after_arcsec":    round(partial_after_arcsec, 3),
        # saturation
        "peak_electrons":          round(peak_e, 1) if not np.isnan(peak_e) else np.nan,
        "sat_status":              status,
        # pixel losses
        "pixel_loss_psf":          round(pl_psf, 2),
        "pixel_loss_saturated":    round(pl_saturated, 2),
        "pixel_loss_unsaturated":  round(pl_unsaturated, 2),
        "pixel_loss_recommended":  round(pl_recommended, 2),
        # focal-plane endpoints (needed for Shapely union in File 2)
        "_x0": x0, "_y0": y0, "_x1": x1, "_y1": y1,
        "_W_mask": W_mask,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Process ONE pointing → File 2 row
# ─────────────────────────────────────────────────────────────────────────────

def process_pointing(pid, streak_rows, pt_meta):
    """
    streak_rows : list of dicts from process_streak (File 1 rows for this pointing)
    pt_meta     : dict with pointing_ra, pointing_dec, pointing_filter,
                  pointing_filter_raw, pointing_exptime
    Returns File 2 dict.
    """
    n_streaks = len(streak_rows)

    # Count statuses
    statuses = [r["sat_status"] for r in streak_rows]
    n_saturated   = statuses.count("saturated")
    n_unsaturated = statuses.count("unsaturated")

    # ── Build Shapely polygons and compute union areas ─────────────────────
    psf_polys  = []
    rec_polys  = []

    for r in streak_rows:
        x0, y0 = r["_x0"], r["_y0"]
        x1, y1 = r["_x1"], r["_y1"]
        W       = r["_W_mask"]

        poly_psf = streak_polygon(x0, y0, x1, y1, PSF_FWHM_PX / 2.0)
        poly_rec = streak_polygon(x0, y0, x1, y1, W / 2.0)

        if poly_psf and poly_psf.is_valid:
            psf_polys.append(poly_psf)
        if poly_rec and poly_rec.is_valid:
            rec_polys.append(poly_rec)

    # Union area in pixels²  (shapely area = px² since coords are in pixels)
    psf_union_area = unary_union(psf_polys).area if psf_polys else 0.0
    rec_union_area = unary_union(rec_polys).area if rec_polys else 0.0

    psf_fraction = psf_union_area / TOTAL_PIXELS
    rec_fraction = rec_union_area / TOTAL_PIXELS

    return {
        "pointing_id":                         pid,
        "pointing_ra":                         round(pt_meta["pointing_ra"], 5),
        "pointing_dec":                        round(pt_meta["pointing_dec"], 5),
        "pointing_filter":                     pt_meta["pointing_filter"],
        "pointing_filter_raw":                 pt_meta["pointing_filter_raw"],
        "pointing_exptime":                    pt_meta["pointing_exptime"],
        "n_streaks":                           n_streaks,
        "total_pixel_loss_psf_fraction":       round(psf_fraction, 8),
        "total_pixel_loss_recommended_fraction": round(rec_fraction, 8),
        "n_saturated":                         n_saturated,
        "n_unsaturated":                       n_unsaturated,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Worker — processes ONE day
# ─────────────────────────────────────────────────────────────────────────────

FILE1_COLS = [
    "pointing_id", "sat_name", "shell_id",
    "pointing_filter", "pointing_filter_raw", "pointing_exptime",
    "ab_magnitude", "streak_type", "ang_vel_arcsec_s", "L_px",
    "partial_before_arcsec", "partial_after_arcsec",
    "peak_electrons", "sat_status",
    "pixel_loss_psf",
    "pixel_loss_saturated", "pixel_loss_unsaturated",
    "pixel_loss_recommended",
]

FILE2_COLS = [
    "pointing_id", "pointing_ra", "pointing_dec",
    "pointing_filter", "pointing_filter_raw",
    "pointing_exptime", "n_streaks",
    "total_pixel_loss_psf_fraction",
    "total_pixel_loss_recommended_fraction",
    "n_saturated", "n_unsaturated",
]


def process_day(
    step2_path:  str,
    azel_path:   str,
    bright_path: str,
    output_dir:  str,
    log_path:    str,
):
    setup_logging(output_dir, os.path.basename(log_path))

    date_str = (os.path.basename(step2_path)
                .replace("streak_trajectories_", "")
                .replace(".csv", ""))

    f1_path = os.path.join(output_dir, f"step5_datapoints_{date_str}.csv")
    f2_path = os.path.join(output_dir, f"step5_pointings_{date_str}.csv")

    t0 = time.perf_counter()
    log(f"[START] {date_str}")

    # ── Resume check ───────────────────────────────────────────────────────
    if os.path.isfile(f1_path) and os.path.isfile(f2_path):
        log(f"[SKIP]  {date_str} — output files exist")
        return {"date": date_str, "status": "skipped"}

    # ── Load step2 ─────────────────────────────────────────────────────────
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

    # ── Attach azel ────────────────────────────────────────────────────────
    if os.path.isfile(azel_path):
        try:
            azel = pd.read_csv(azel_path, usecols=["az_deg", "el_deg"])
            if len(azel) == N:
                df["az_deg"] = azel["az_deg"].values
                df["el_deg"] = azel["el_deg"].values
            else:
                log(f"[WARN]  {date_str} — azel row mismatch ({len(azel)} vs {N})")
        except Exception as e:
            log(f"[WARN]  {date_str} — azel read failed: {e}")
    else:
        log(f"[WARN]  {date_str} — azel file missing: {azel_path}")

    # ── Attach brightness ──────────────────────────────────────────────────
    if os.path.isfile(bright_path):
        try:
            bright = pd.read_csv(bright_path, usecols=["ab_magnitude"])
            if len(bright) == N:
                df["ab_magnitude"] = bright["ab_magnitude"].values
            else:
                log(f"[WARN]  {date_str} — bright row mismatch ({len(bright)} vs {N})")
                df["ab_magnitude"] = np.nan
        except Exception as e:
            log(f"[WARN]  {date_str} — bright read failed: {e}")
            df["ab_magnitude"] = np.nan
    else:
        log(f"[WARN]  {date_str} — bright file missing: {bright_path}")
        df["ab_magnitude"] = np.nan

    log(f"  {date_str} | {N:,} rows loaded")

    # ── Process streaks ────────────────────────────────────────────────────
    file1_rows   = []
    pointing_map = {}   # pid → {"meta": ..., "streaks": [...]}

    grouped = df.groupby(["pointing_id", "sat_name", "shell_id"])
    n_groups = len(grouped)
    log(f"  {date_str} | {n_groups:,} streak groups to process")

    for i, ((pid, sat, shell), grp) in enumerate(grouped):
        pt_row   = grp.iloc[0]
        pt_ra    = float(pt_row["pointing_ra"])
        pt_dec   = float(pt_row["pointing_dec"])
        filt_raw = str(pt_row["pointing_filter"])   # e.g. "r_57"
        filt     = base_band(filt_raw)               # e.g. "r"
        t_exp    = float(pt_row["pointing_exptime"])

        result = process_streak(grp, pt_ra, pt_dec, filt, filt_raw, t_exp)
        if result is None:
            continue

        # Stash File 1 row (without internal focal-plane keys)
        f1_row = {k: v for k, v in result.items() if not k.startswith("_")}
        file1_rows.append(f1_row)

        # Accumulate for File 2
        if pid not in pointing_map:
            pointing_map[pid] = {
                "meta": {
                    "pointing_ra":  pt_ra,
                    "pointing_dec": pt_dec,
                    "pointing_filter": filt,
                    "pointing_filter_raw": filt_raw,
                    "pointing_exptime": t_exp,
                },
                "streaks": [],
            }
        pointing_map[pid]["streaks"].append(result)

        if (i + 1) % 500 == 0 or i == n_groups - 1:
            log(f"  {date_str} | streaks {i+1:,}/{n_groups:,}")

    # ── Save File 1 ────────────────────────────────────────────────────────
    f1_df = pd.DataFrame(file1_rows, columns=FILE1_COLS)
    f1_df.to_csv(f1_path, index=False)
    log(f"  {date_str} | File1 saved: {len(f1_df):,} streaks  "
        f"({os.path.getsize(f1_path)/1e6:.2f} MB)")

    # ── Process pointings (File 2) ─────────────────────────────────────────
    file2_rows = []
    n_pts = len(pointing_map)
    log(f"  {date_str} | building Shapely unions for {n_pts:,} pointings …")

    for j, (pid, pdata) in enumerate(pointing_map.items()):
        f2_row = process_pointing(pid, pdata["streaks"], pdata["meta"])
        file2_rows.append(f2_row)

        if (j + 1) % 200 == 0 or j == n_pts - 1:
            log(f"  {date_str} | pointings {j+1:,}/{n_pts:,}")

    f2_df = pd.DataFrame(file2_rows, columns=FILE2_COLS)
    f2_df.to_csv(f2_path, index=False)
    log(f"  {date_str} | File2 saved: {len(f2_df):,} pointings  "
        f"({os.path.getsize(f2_path)/1e6:.2f} MB)")

    elapsed = time.perf_counter() - t0
    log(f"[DONE]  {date_str} | {_hms(elapsed)}")

    del df, f1_df, f2_df
    gc.collect()

    return {
        "date":        date_str,
        "status":      "done",
        "n_streaks":   len(file1_rows),
        "n_pointings": n_pts,
        "elapsed":     elapsed,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Build / append daily summary (File 3)
# ─────────────────────────────────────────────────────────────────────────────

FILE3_COLS = [
    "date", "pointing_night",
    "n_pointings", "n_affected_pointings",
    "mean_pixel_loss_psf_fraction", "max_pixel_loss_psf_fraction",
    "mean_pixel_loss_recommended_fraction", "max_pixel_loss_recommended_fraction",
    "total_saturated", "total_unsaturated",
]


def append_daily_summary(output_dir: str, date_str: str,
                          step2_path: str):
    """
    Reads File 2 for date_str, computes daily aggregates, appends to File 3.
    """
    f2_path = os.path.join(output_dir, f"step5_pointings_{date_str}.csv")
    f3_path = os.path.join(output_dir, "step5_daily_summary.csv")

    if not os.path.isfile(f2_path):
        return

    f2 = pd.read_csv(f2_path)

    # Get pointing_night from step2
    try:
        night = int(pd.read_csv(step2_path, usecols=["pointing_night"],
                                nrows=1)["pointing_night"].iloc[0])
    except Exception:
        night = -1

    n_total    = len(f2)
    n_affected = int((f2["n_streaks"] > 0).sum())

    row = {
        "date":                                date_str,
        "pointing_night":                      night,
        "n_pointings":                         n_total,
        "n_affected_pointings":                n_affected,
        "mean_pixel_loss_psf_fraction":        round(f2["total_pixel_loss_psf_fraction"].mean(), 8),
        "max_pixel_loss_psf_fraction":         round(f2["total_pixel_loss_psf_fraction"].max(), 8),
        "mean_pixel_loss_recommended_fraction":round(f2["total_pixel_loss_recommended_fraction"].mean(), 8),
        "max_pixel_loss_recommended_fraction": round(f2["total_pixel_loss_recommended_fraction"].max(), 8),
        "total_saturated":                     int(f2["n_saturated"].sum()),
        "total_unsaturated":                   int(f2["n_unsaturated"].sum()),
    }

    write_header = not os.path.isfile(f3_path)
    pd.DataFrame([row], columns=FILE3_COLS).to_csv(
        f3_path, mode="a", header=write_header, index=False
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(
    step2_dir    = "step2_output",
    step3_dir    = "step3_output",
    step4_dir    = "step4_output_20deg",
    output_dir   = "step5_output",
    n_workers    = 40,
    log_filename = "step5.log",
):
    log_path   = setup_logging(output_dir, log_filename)
    wall_start = time.perf_counter()

    log("=" * 70)
    log("  STEP 5: Pixel Loss  (parallel days)")
    log("=" * 70)
    log(f"  step2_dir  : {step2_dir}")
    log(f"  step3_dir  : {step3_dir}")
    log(f"  step4_dir  : {step4_dir}")
    log(f"  output_dir : {output_dir}")
    log(f"  n_workers  : {n_workers}")

    os.makedirs(output_dir, exist_ok=True)

    # ── Collect day CSVs ───────────────────────────────────────────────────
    all_step2 = sorted(glob.glob(
        os.path.join(step2_dir, "streak_trajectories_*.csv")
    ))
    all_step2 = [p for p in all_step2 if "_azel" not in p and "_bright" not in p]

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

    # ── Build worker args ──────────────────────────────────────────────────
    worker_args = []
    for step2_path in all_step2:
        date_str    = (os.path.basename(step2_path)
                       .replace("streak_trajectories_", "")
                       .replace(".csv", ""))
        azel_path   = os.path.join(
            step3_dir, f"streak_trajectories_{date_str}_azel.csv"
        )
        bright_path = os.path.join(
            step4_dir, f"streak_trajectories_{date_str}_bright.csv"
        )
        worker_args.append((
            step2_path, azel_path, bright_path,
            output_dir, log_path,
        ))

    log(f"[INFO] Dispatching {len(worker_args)} days to {n_workers} workers …\n")

    # ── Parallel dispatch ──────────────────────────────────────────────────
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(process_day, worker_args)

    # ── Append daily summaries ─────────────────────────────────────────────
    log("\n[INFO] Building daily summary (File 3) …")
    for step2_path, *_ in worker_args:
        date_str = (os.path.basename(step2_path)
                    .replace("streak_trajectories_", "")
                    .replace(".csv", ""))
        append_daily_summary(output_dir, date_str, step2_path)

    # ── Final summary ──────────────────────────────────────────────────────
    log("")
    log("=" * 70)
    log("  RESULTS SUMMARY")
    log("=" * 70)

    n_done    = sum(1 for r in results if r["status"] == "done")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_errors  = sum(1 for r in results if r["status"] == "error")

    log(f"  Done    : {n_done}")
    log(f"  Skipped : {n_skipped}")
    log(f"  Errors  : {n_errors}")

    if n_done > 0:
        done = [r for r in results if r["status"] == "done"]
        log(f"  Avg time/day : {_hms(sum(r['elapsed'] for r in done)/n_done)}")

    f3_path = os.path.join(output_dir, "step5_daily_summary.csv")
    if os.path.isfile(f3_path):
        f3 = pd.read_csv(f3_path)
        log(f"\n  Daily summary ({len(f3)} days):")
        log(f"  Mean pixel loss PSF        : {f3['mean_pixel_loss_psf_fraction'].mean():.6f}")
        log(f"  Mean pixel loss recommended: {f3['mean_pixel_loss_recommended_fraction'].mean():.6f}")
        log(f"  Max  pixel loss recommended: {f3['max_pixel_loss_recommended_fraction'].max():.6f}")
        log(f"  Total saturated streaks    : {f3['total_saturated'].sum():,}")
        log(f"  Total unsaturated streaks  : {f3['total_unsaturated'].sum():,}")

    log(f"\n  Total wall time: {_hms(time.perf_counter() - wall_start)}")
    log("[DONE]")
    log(f"[LOG]   {log_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(
        step2_dir  = "step2_output",
        step3_dir  = "step3_output",
        step4_dir  = "step4_output_20deg_rad",
        output_dir = "step5_output_pixel_loss_rad",
        n_workers  = 60,
    )