"""
step5_combined_loss.py  (v4 — with correct RotSkyPos camera orientation)
─────────────────────────────────────────────────────────────────────────────
Computes combined pixel loss for pointings that contain at least one
satellite streak with AB magnitude < MAG_LIMIT.

ORIENTATION FIX (v4)
────────────────────
The gnomonic projection (radec_to_focal_px) maps sky coords to a frame
where +Y = North, +X = East.  However the physical camera is rotated on
the sky by RotSkyPos degrees (East of North) for each pointing.

From SMTN-019:
  RotSkyPos = orientation of +Y_DVCS measured East of North (ICRF)
  Odd number of mirrors in SST causes an additional 180° flip.

Therefore to convert from the sky-North frame to the camera/CCD frame:
  angle = 180° - RotSkyPos
  x_cam =  cos(angle)*x_sky + sin(angle)*y_sky
  y_cam = -sin(angle)*x_sky + cos(angle)*y_sky

RotSkyPos is read from the opsim database via a lookup table built
at the start of each day's processing.

FILTER SUFFIX FIX (this version)
─────────────────────────────────
    pointing_filter in step2 is the raw opsim value (band letter + filter
    load/throughput-curve id, e.g. "r_57", "g_12") — NOT a bare single
    letter. The original code used this raw value directly as the `band`
    key into BAND_ZP / sb_to_peak_electrons() for NORMAL streaks
    (ab_mag >= MAG_LIMIT), which only has plain single-letter keys
    ('u','g','r','i','z','y'). Since "r_57" is never a member of BAND_ZP,
    sb_to_peak_electrons() always returned NaN regardless of actual
    brightness, so classify_status() always returned "faint" for every
    normal streak — saturated/unsaturated counts were always 0 across the
    entire survey. (BRIGHT streaks, ab_mag < MAG_LIMIT, were unaffected —
    they never call classify_status at all, since the whole CCD is marked
    dead outright.) Fixed by stripping the suffix (str.split("_")[0]) at
    the point pointing_filter is read, exactly as step5_pixel_loss.py's
    base_band() helper already does.

WORKFLOW
────────
1.  Build pointing_rot lookup: pointing_id → rotSkyPos from opsim DB
2.  Find all pointings with at least one streak where ab_mag < MAG_LIMIT
3.  For EVERY streak in those pointings:
      a) Map streak path → focal plane px WITH rotation applied
      b) Identify CCD IDs hit
4.  Pixel loss per pointing:
      BRIGHT streaks  (ab_mag < MAG_LIMIT):
        → all CCDs touched are marked DEAD
      NORMAL streaks  (ab_mag >= MAG_LIMIT):
        → classify: saturated / unsaturated / faint
        → build streak polygon, subtract dead CCDs, add to Shapely union
5.  Total pixel loss = dead_px + normal_px (no overlap by construction)

INPUTS
──────
  step2_output/streak_trajectories_YYYY-MM-DD.csv
  step4_output_20deg/streak_trajectories_YYYY-MM-DD_bright.csv
  step6_output_perband/step6_YYYY-MM-DD_{band}.csv
  opsim database (for rotSkyPos per pointing_id)

OUTPUTS
───────
  step5_combined_loss_output_mag7/combined_datapoints_YYYY-MM-DD.csv
  step5_combined_loss_output_mag7/combined_pointings_YYYY-MM-DD.csv
  step5_combined_loss_output_mag7/combined_daily_summary.csv
"""

import os, gc, glob, time, logging, sys, math, sqlite3
from multiprocessing import Pool

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, box as shapely_box
from shapely.ops import unary_union

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str, log_filename: str) -> str:
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
# !! MAGNITUDE THRESHOLD !!
# ─────────────────────────────────────────────────────────────────────────────
MAG_LIMIT = 2.0   # streaks with ab_mag < MAG_LIMIT → whole CCD lost

# ─────────────────────────────────────────────────────────────────────────────
# Camera geometry  (confirmed config)
# ─────────────────────────────────────────────────────────────────────────────

CCD_SIZE_MM   = 42.0
GAP_CCD_MM    = 0.27
GAP_RAFT_MM   = 0.50
RAFT_SIZE_MM  = 3 * CCD_SIZE_MM + 2 * GAP_CCD_MM    # 126.54 mm
STEP_MM       = RAFT_SIZE_MM + GAP_RAFT_MM           # 127.04 mm
TOTAL_SPAN_MM = 5 * STEP_MM - GAP_RAFT_MM            # 635.20 mm
FOV_RADIUS_MM = TOTAL_SPAN_MM / 2                    # 317.60 mm
FOV_DEG       = 1.75
MM_PER_DEG    = FOV_RADIUS_MM / FOV_DEG
DEG_PER_MM    = FOV_DEG / FOV_RADIUS_MM

PLATE_SCALE_ARCSEC_PX = 0.2
PX_PER_DEG    = 3600.0 / PLATE_SCALE_ARCSEC_PX       # 18000 px/deg
PX_PER_MM     = PX_PER_DEG * DEG_PER_MM

CCD_SIZE_PX_PHYS = CCD_SIZE_MM * PX_PER_MM
GAP_CCD_PX       = GAP_CCD_MM  * PX_PER_MM
STEP_PX          = STEP_MM * PX_PER_MM
RAFT_SIZE_PX     = RAFT_SIZE_MM * PX_PER_MM
FOV_RADIUS_PX    = FOV_RADIUS_MM * PX_PER_MM

CCD_PX        = 4096
CCD_PIXELS    = CCD_PX * CCD_PX                      # 16,777,216
TOTAL_PIXELS  = 3.2e9
FOV_RADIUS_DEG = FOV_DEG

# Physics
PSF_FWHM_ARCSEC    = 0.67
PSF_FWHM_PX        = PSF_FWHM_ARCSEC / PLATE_SCALE_ARCSEC_PX
PSF_PEAK_FRAC      = 0.44
PIXEL_AREA_ARCSEC2 = PLATE_SCALE_ARCSEC_PX ** 2
FULL_WELL_E        = 130_000

BAND_ZP = {
    'u': 26.52, 'g': 28.51, 'r': 28.36,
    'i': 28.17, 'z': 27.78, 'y': 26.82,
}
MASK_WIDTH_PX = {
    "saturated":   500,
    "unsaturated": 100,
    "faint":       0,
}

# ─────────────────────────────────────────────────────────────────────────────
# Build CCD grid
# ─────────────────────────────────────────────────────────────────────────────

GRID_TYPE = [
    [ 'WF', 'ITL', 'ITL', 'ITL',  'WF'],
    ['e2V', 'e2V', 'e2V', 'e2V', 'e2V'],
    ['ITL', 'e2V', 'e2V', 'e2V', 'e2V'],
    ['ITL', 'e2V', 'e2V', 'e2V', 'e2V'],
    [ 'WF', 'ITL', 'ITL', 'ITL',  'WF'],
]

RAFT_NAMES = [
    ['R40','R41','R42','R43','R44'],
    ['R30','R31','R32','R33','R34'],
    ['R20','R21','R22','R23','R24'],
    ['R10','R11','R12','R13','R14'],
    ['R00','R01','R02','R03','R04'],
]

CCD_BBOX_PX = {}   # ccd_id → (x_min, x_max, y_min, y_max) in camera-frame px
CCD_ID_MAP  = {}   # (rr, rc, cr, cc) → ccd_id

ccd_n = 1
for rr in range(5):
    for rc in range(5):
        if GRID_TYPE[rr][rc] not in ('ITL', 'e2V'):
            continue
        raft_x_mm = -FOV_RADIUS_MM + rc * STEP_MM
        raft_y_mm =  FOV_RADIUS_MM - rr * STEP_MM - RAFT_SIZE_MM
        for cr in range(3):
            for cc in range(3):
                ccd_x_mm = raft_x_mm + cc * (CCD_SIZE_MM + GAP_CCD_MM)
                ccd_y_mm = raft_y_mm + (2 - cr) * (CCD_SIZE_MM + GAP_CCD_MM)
                x0 = ccd_x_mm * PX_PER_MM
                y0 = ccd_y_mm * PX_PER_MM
                x1 = x0 + CCD_SIZE_PX_PHYS
                y1 = y0 + CCD_SIZE_PX_PHYS
                CCD_BBOX_PX[ccd_n]        = (x0, x1, y0, y1)
                CCD_ID_MAP[(rr,rc,cr,cc)] = ccd_n
                ccd_n += 1

N_SCIENCE_CCDS = ccd_n - 1  # 189

_CCD_IDS   = np.array(list(CCD_BBOX_PX.keys()),   dtype=np.int16)
_CCD_BOXES = np.array(list(CCD_BBOX_PX.values()),  dtype=np.float64)  # (189,4)

def ccd_shapely_box(ccd_id):
    x0, x1, y0, y1 = CCD_BBOX_PX[ccd_id]
    return shapely_box(x0, y0, x1, y1)

# ─────────────────────────────────────────────────────────────────────────────
# opsim rotSkyPos lookup
# ─────────────────────────────────────────────────────────────────────────────

def load_rot_sky_pos(opsim_db_path: str) -> dict:
    """
    Returns dict: pointing_id (int) → rotSkyPos (float, degrees)
    Loaded once in main, passed to workers.
    """
    with sqlite3.connect(opsim_db_path) as con:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", con
        )["name"].tolist()
        table = next((t for t in ["observations","SummaryAllProps","Summary"]
                      if t in tables), None)
        df = pd.read_sql(
            f"SELECT observationId, rotSkyPos FROM {table}", con
        )
    return dict(zip(df["observationId"].astype(int),
                    df["rotSkyPos"].astype(float)))

# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def radec_to_focal_px(ra_deg, dec_deg, ra0_deg, dec0_deg, rot_sky_pos_deg=0.0):
    """
    Gnomonic projection of (ra_deg, dec_deg) onto focal-plane pixels,
    then rotate by camera orientation.

    From SMTN-019:
      RotSkyPos = orientation of +Y_DVCS East of North (ICRF)
      Odd number of mirrors adds 180° flip.
      So total rotation from sky-North frame to camera frame:
        angle = 180° - RotSkyPos

    Parameters
    ----------
    rot_sky_pos_deg : float
        rotSkyPos for this pointing (degrees), from opsim DB.
        Default 0.0 = no rotation (North up).
    """
    ra  = np.radians(ra_deg);   dec  = np.radians(dec_deg)
    ra0 = np.radians(ra0_deg);  dec0 = np.radians(dec0_deg)

    # Gnomonic projection → sky-North frame (North=+Y, East=+X)
    cos_c = (np.sin(dec0)*np.sin(dec) +
             np.cos(dec0)*np.cos(dec)*np.cos(ra - ra0))
    x_rad = np.cos(dec)*np.sin(ra - ra0) / cos_c
    y_rad = (np.cos(dec0)*np.sin(dec) -
             np.sin(dec0)*np.cos(dec)*np.cos(ra - ra0)) / cos_c
    x_sky = np.degrees(x_rad) * PX_PER_DEG
    y_sky = np.degrees(y_rad) * PX_PER_DEG

    # Rotate to camera/CCD frame
    # angle = 180° - RotSkyPos  (SMTN-019: mirror flip + sky rotation)
    angle_rad = np.radians(180.0 - rot_sky_pos_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x_cam =  cos_a * x_sky + sin_a * y_sky
    y_cam = -sin_a * x_sky + cos_a * y_sky

    return x_cam, y_cam


def angular_sep_deg(ra1, dec1, ra2, dec2):
    r1 = np.radians(ra1); d1 = np.radians(dec1)
    r2 = np.radians(ra2); d2 = np.radians(dec2)
    cos_a = (np.sin(d1)*np.sin(d2) +
             np.cos(d1)*np.cos(d2)*np.cos(r1 - r2))
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def interp_fov_boundary(ra_in, dec_in, ra_out, dec_out, pt_ra, pt_dec):
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid   = 0.5*(lo + hi)
        ra_m  = ra_in  + mid*(ra_out  - ra_in)
        dec_m = dec_in + mid*(dec_out - dec_in)
        if angular_sep_deg(pt_ra, pt_dec, ra_m, dec_m) < FOV_RADIUS_DEG:
            lo = mid
        else:
            hi = mid
    f = 0.5*(lo + hi)
    return ra_in + f*(ra_out - ra_in), dec_in + f*(dec_out - dec_in)

# ─────────────────────────────────────────────────────────────────────────────
# CCD hit detection  (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def point_to_ccds(x_px, y_px):
    mask = (
        (x_px >= _CCD_BOXES[:, 0]) & (x_px <= _CCD_BOXES[:, 1]) &
        (y_px >= _CCD_BOXES[:, 2]) & (y_px <= _CCD_BOXES[:, 3])
    )
    return set(_CCD_IDS[mask].tolist())


def segment_to_ccds(x0, y0, x1, y1, n_interp=50):
    ts = np.linspace(0.0, 1.0, n_interp + 2)
    xs = x0 + ts*(x1 - x0)
    ys = y0 + ts*(y1 - y0)
    hit = set()
    for x, y in zip(xs, ys):
        hit |= point_to_ccds(x, y)
    return hit

# ─────────────────────────────────────────────────────────────────────────────
# Streak waypoints  (Option A + B) — now takes rot_sky_pos
# ─────────────────────────────────────────────────────────────────────────────

def streak_waypoints_px(grp, pt_ra, pt_dec, rot_sky_pos_deg=0.0):
    """
    Returns arrays (wp_x, wp_y) in camera-frame pixels (rotation applied).
    Includes interpolated FOV boundary points for entering/exiting streaks.
    """
    fov    = grp['in_fov'].values
    in_idx = np.where(fov)[0]
    if len(in_idx) == 0:
        return None, None

    ra_arr  = grp['ra_deg'].values
    dec_arr = grp['dec_deg'].values
    first_in = in_idx[0];  last_in = in_idx[-1]
    entering = first_in > 0
    exiting  = last_in  < len(grp) - 1

    wp_ra, wp_dec = [], []

    if entering:
        r_bnd, d_bnd = interp_fov_boundary(
            ra_arr[first_in],   dec_arr[first_in],
            ra_arr[first_in-1], dec_arr[first_in-1],
            pt_ra, pt_dec)
        wp_ra.append(r_bnd);  wp_dec.append(d_bnd)

    for i in in_idx:
        wp_ra.append(ra_arr[i]);  wp_dec.append(dec_arr[i])

    if exiting:
        r_bnd, d_bnd = interp_fov_boundary(
            ra_arr[last_in],   dec_arr[last_in],
            ra_arr[last_in+1], dec_arr[last_in+1],
            pt_ra, pt_dec)
        wp_ra.append(r_bnd);  wp_dec.append(d_bnd)

    # ── Apply rotation here ───────────────────────────────────────────────
    wp_x, wp_y = radec_to_focal_px(
        np.array(wp_ra), np.array(wp_dec),
        pt_ra, pt_dec,
        rot_sky_pos_deg=rot_sky_pos_deg   # ← camera orientation
    )
    return wp_x, wp_y


def streak_to_ccds(wp_x, wp_y):
    hit = set()
    for k in range(len(wp_x)):
        hit |= point_to_ccds(wp_x[k], wp_y[k])
    for k in range(len(wp_x) - 1):
        hit |= segment_to_ccds(wp_x[k], wp_y[k], wp_x[k+1], wp_y[k+1])
    return hit

# ─────────────────────────────────────────────────────────────────────────────
# Streak polygon
# ─────────────────────────────────────────────────────────────────────────────

def make_streak_polygon(wp_x, wp_y, half_width_px):
    polys = []
    for k in range(len(wp_x) - 1):
        x0, y0 = wp_x[k],   wp_y[k]
        x1, y1 = wp_x[k+1], wp_y[k+1]
        dx = x1 - x0;  dy = y1 - y0
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        nx = -dy/length;  ny = dx/length
        corners = [
            (x0 + nx*half_width_px, y0 + ny*half_width_px),
            (x1 + nx*half_width_px, y1 + ny*half_width_px),
            (x1 - nx*half_width_px, y1 - ny*half_width_px),
            (x0 - nx*half_width_px, y0 - ny*half_width_px),
        ]
        p = Polygon(corners)
        if p.is_valid and not p.is_empty:
            polys.append(p)
    if not polys:
        return None
    result = unary_union(polys)
    return result if not result.is_empty else None

# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────

def sb_to_peak_electrons(sb, band, t_crossing_sec):
    if np.isnan(sb) or np.isnan(t_crossing_sec) or t_crossing_sec <= 0:
        return np.nan
    if band not in BAND_ZP:
        return np.nan
    zp               = BAND_ZP[band]
    flux_per_arcsec2 = 10.0 ** ((zp - sb) / 2.5)
    e_per_px_per_sec = flux_per_arcsec2 * PIXEL_AREA_ARCSEC2
    return e_per_px_per_sec * t_crossing_sec * PSF_PEAK_FRAC


def classify_status(sb, band, t_crossing_sec):
    if np.isnan(sb):
        return "faint", np.nan
    peak_e = sb_to_peak_electrons(sb, band, t_crossing_sec)
    if np.isnan(peak_e):
        return "faint", np.nan
    status = "saturated" if peak_e > FULL_WELL_E else "unsaturated"
    return status, peak_e

# ─────────────────────────────────────────────────────────────────────────────
# Column definitions
# ─────────────────────────────────────────────────────────────────────────────

DP_COLS = [
    "pointing_id", "sat_name", "shell_id",
    "pointing_filter", "pointing_exptime", "pointing_night",
    "rot_sky_pos",                         # ← new: camera orientation
    "min_ab_magnitude", "streak_type", "streak_category",
    "sat_status",
    "n_ccds_hit", "ccd_ids_hit",
    "streak_pixel_loss_raw",
    "streak_pixel_loss_clipped",
]

PT_COLS = [
    "pointing_id", "pointing_ra", "pointing_dec",
    "pointing_filter", "pointing_exptime", "pointing_night",
    "rot_sky_pos",                         # ← new
    "n_bright_streaks", "n_normal_streaks",
    "n_ccds_dead", "dead_ccd_ids",
    "dead_pixel_loss",
    "normal_pixel_loss_clipped",
    "total_pixel_loss",
    "total_pixel_loss_fraction",
]

SUMMARY_COLS = [
    "date", "pointing_night",
    "n_affected_pointings",
    "total_bright_streaks", "total_normal_streaks",
    "mean_ccds_dead_per_pointing", "max_ccds_dead_per_pointing",
    "mean_total_pixel_loss_fraction", "max_total_pixel_loss_fraction",
]

# ─────────────────────────────────────────────────────────────────────────────
# Worker: process ONE day
# ─────────────────────────────────────────────────────────────────────────────

def process_day(args):
    step2_path, bright_path, step6_dir, output_dir, log_path, rot_lookup = args
    setup_logging(output_dir, os.path.basename(log_path))

    date_str = (os.path.basename(step2_path)
                .replace("streak_trajectories_", "").replace(".csv", ""))

    f1_path = os.path.join(output_dir, f"combined_datapoints_{date_str}.csv")
    f2_path = os.path.join(output_dir, f"combined_pointings_{date_str}.csv")

    if os.path.isfile(f1_path) and os.path.isfile(f2_path):
        log(f"[SKIP]  {date_str}")
        return {"date": date_str, "status": "skipped"}

    t0 = time.perf_counter()
    log(f"[START] {date_str}")

    # ── Load step2 ────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(step2_path, usecols=[
            "pointing_id", "pointing_ra", "pointing_dec",
            "pointing_filter", "pointing_exptime", "pointing_night",
            "shell_id", "sat_name", "step",
            "ra_deg", "dec_deg", "in_fov", "sunlit",
        ])
    except Exception as e:
        log(f"[ERROR] {date_str} step2: {e}")
        return {"date": date_str, "status": "error"}

    N = len(df)
    log(f"  {date_str} | {N:,} step2 rows")

    # ── Load step4 brightness (row-aligned) ───────────────────────────────
    if not os.path.isfile(bright_path):
        log(f"[ERROR] {date_str} bright file missing: {bright_path}")
        return {"date": date_str, "status": "error"}

    try:
        bright = pd.read_csv(bright_path, usecols=["ab_magnitude_corrected"])
    except Exception as e:
        log(f"[ERROR] {date_str} bright: {e}")
        return {"date": date_str, "status": "error"}

    if len(bright) != N:
        log(f"[ERROR] {date_str} row mismatch step2={N} bright={len(bright)}")
        return {"date": date_str, "status": "error"}

    df["ab_magnitude"] = bright["ab_magnitude_corrected"].values
    del bright; gc.collect()

    # ── Step 1: Find pointings with at least one ab_mag < MAG_LIMIT ───────
    GROUP_KEYS = ["pointing_id", "sat_name", "shell_id"]

    valid_mask = df["in_fov"] & df["sunlit"] & df["ab_magnitude"].notna()
    df_valid   = df[valid_mask]

    if df_valid.empty:
        log(f"  {date_str} | no valid rows → skip")
        return {"date": date_str, "status": "done_empty"}

    min_ab = (df_valid.groupby(GROUP_KEYS)["ab_magnitude"]
              .min().reset_index()
              .rename(columns={"ab_magnitude": "min_ab_mag"}))

    bright_pids = set(
        min_ab.loc[min_ab["min_ab_mag"] < MAG_LIMIT, "pointing_id"].unique()
    )
    log(f"  {date_str} | {len(bright_pids):,} pointings with ab_mag < {MAG_LIMIT}")

    if not bright_pids:
        return {"date": date_str, "status": "done_no_bright"}

    # ── Filter to affected pointings ──────────────────────────────────────
    df_affected = df[df["pointing_id"].isin(bright_pids)].copy()
    df_affected = df_affected.merge(min_ab, on=GROUP_KEYS, how="left")
    log(f"  {date_str} | {len(df_affected):,} rows in affected pointings")

    # ── Load step6 SB lookup ──────────────────────────────────────────────
    sb_lookup = {}
    for band in "ugrizy":
        s6_path = os.path.join(step6_dir, f"step6_{date_str}_{band}.csv")
        if not os.path.exists(s6_path):
            continue
        try:
            s6 = pd.read_csv(s6_path, usecols=[
                "pointing_id", "sat_name", "shell_id",
                "surface_brightness_mag_arcsec2",
                "t_crossing_sec", "L_px",
            ])
            s6 = s6[s6["pointing_id"].isin(bright_pids)]
        except Exception as e:
            log(f"  [WARN] {date_str} {band}: {e}")
            continue
        for _, row in s6.iterrows():
            key = (int(row["pointing_id"]), row["sat_name"], int(row["shell_id"]))
            sb_lookup[key] = {
                "sb":             row["surface_brightness_mag_arcsec2"],
                "t_crossing_sec": row["t_crossing_sec"],
                "L_px":           row["L_px"],
            }
        del s6; gc.collect()

    log(f"  {date_str} | SB entries loaded: {len(sb_lookup):,}")

    # ── Process streaks ───────────────────────────────────────────────────
    grouped  = df_affected.groupby(GROUP_KEYS)
    n_groups = len(grouped)
    log(f"  {date_str} | {n_groups:,} streak groups to process")

    dp_rows      = []
    pointing_map = {}

    for i, ((pid, sat, shell), grp) in enumerate(grouped):
        grp = grp.sort_values("step").reset_index(drop=True)

        pt_ra  = float(grp["pointing_ra"].iloc[0])
        pt_dec = float(grp["pointing_dec"].iloc[0])
        filt   = base_band(grp["pointing_filter"].iloc[0])   # FIX: strip "_NN" suffix, e.g. "r_57" -> "r"
        t_exp  = float(grp["pointing_exptime"].iloc[0])
        night  = int(grp["pointing_night"].iloc[0])
        min_ab_val = float(grp["min_ab_mag"].iloc[0]) \
                     if "min_ab_mag" in grp.columns else np.nan

        # ── Get rotSkyPos for this pointing ───────────────────────────────
        rot_sky_pos = rot_lookup.get(int(pid), 0.0)

        if pid not in pointing_map:
            pointing_map[pid] = {
                "meta": {
                    "pointing_ra":      pt_ra,
                    "pointing_dec":     pt_dec,
                    "pointing_filter":  filt,
                    "pointing_exptime": t_exp,
                    "pointing_night":   night,
                    "rot_sky_pos":      rot_sky_pos,
                },
                "dead_ccds":        set(),
                "normal_polys_rec": [],
                "n_bright":         0,
                "n_normal":         0,
            }

        fov    = grp["in_fov"].values
        in_idx = np.where(fov)[0]
        if len(in_idx) == 0:
            continue

        first_in = in_idx[0]; last_in = in_idx[-1]
        entering = first_in > 0;  exiting = last_in < len(grp) - 1
        if   entering and exiting: stype = "both"
        elif entering:             stype = "entering"
        elif exiting:              stype = "exiting"
        else:                      stype = "full"

        # ── Build waypoints WITH rotation ─────────────────────────────────
        wp_x, wp_y = streak_waypoints_px(
            grp, pt_ra, pt_dec,
            rot_sky_pos_deg=rot_sky_pos    # ← camera orientation applied
        )
        if wp_x is None:
            continue

        hit_ccds = streak_to_ccds(wp_x, wp_y)

        # ── BRIGHT streak ─────────────────────────────────────────────────
        if min_ab_val < MAG_LIMIT:
            pointing_map[pid]["dead_ccds"] |= hit_ccds
            pointing_map[pid]["n_bright"]  += 1

            dp_rows.append({
                "pointing_id":               int(pid),
                "sat_name":                  sat,
                "shell_id":                  int(shell),
                "pointing_filter":           filt,
                "pointing_exptime":          t_exp,
                "pointing_night":            night,
                "rot_sky_pos":               round(rot_sky_pos, 4),
                "min_ab_magnitude":          round(min_ab_val, 4),
                "streak_type":               stype,
                "streak_category":           "bright",
                "sat_status":                "bright",
                "n_ccds_hit":                len(hit_ccds),
                "ccd_ids_hit":               str(sorted(hit_ccds)),
                "streak_pixel_loss_raw":     len(hit_ccds) * CCD_PIXELS,
                "streak_pixel_loss_clipped": len(hit_ccds) * CCD_PIXELS,
            })

        # ── NORMAL streak ─────────────────────────────────────────────────
        else:
            key = (int(pid), sat, int(shell))
            s6  = sb_lookup.get(key, {})
            sb             = s6.get("sb",             np.nan)
            t_crossing_sec = s6.get("t_crossing_sec", np.nan)

            status, peak_e = classify_status(sb, filt, t_crossing_sec)
            W = MASK_WIDTH_PX.get(status, 0)

            pointing_map[pid]["n_normal"] += 1

            raw_loss     = 0.0
            clipped_loss = 0.0

            if status != "faint" and W > 0 and len(wp_x) > 1:
                poly_full = make_streak_polygon(wp_x, wp_y, W / 2.0)
                if poly_full is not None:
                    raw_loss = poly_full.area
                    dead_now = pointing_map[pid]["dead_ccds"]
                    if dead_now:
                        dead_union   = unary_union([ccd_shapely_box(c) for c in dead_now])
                        poly_clipped = poly_full.difference(dead_union)
                    else:
                        poly_clipped = poly_full
                    if poly_clipped and not poly_clipped.is_empty:
                        clipped_loss = poly_clipped.area
                        pointing_map[pid]["normal_polys_rec"].append(poly_clipped)

            dp_rows.append({
                "pointing_id":               int(pid),
                "sat_name":                  sat,
                "shell_id":                  int(shell),
                "pointing_filter":           filt,
                "pointing_exptime":          t_exp,
                "pointing_night":            night,
                "rot_sky_pos":               round(rot_sky_pos, 4),
                "min_ab_magnitude":          round(min_ab_val, 4),
                "streak_type":               stype,
                "streak_category":           "normal",
                "sat_status":                status,
                "n_ccds_hit":                len(hit_ccds),
                "ccd_ids_hit":               str(sorted(hit_ccds)),
                "streak_pixel_loss_raw":     round(raw_loss, 2),
                "streak_pixel_loss_clipped": round(clipped_loss, 2),
            })

        if (i + 1) % 500 == 0 or i == n_groups - 1:
            log(f"  {date_str} | streaks {i+1:,}/{n_groups:,}")

    # ── Per-pointing summary ──────────────────────────────────────────────
    pt_rows = []
    for pid, pdata in pointing_map.items():
        meta    = pdata["meta"]
        dead    = pdata["dead_ccds"]
        n_dead  = len(dead)
        dead_px = n_dead * CCD_PIXELS

        normal_polys = pdata.get("normal_polys_rec", [])
        if normal_polys:
            normal_union = unary_union(normal_polys)
            if dead:
                dead_union   = unary_union([ccd_shapely_box(c) for c in dead])
                normal_union = normal_union.difference(dead_union)
            normal_px = normal_union.area if not normal_union.is_empty else 0.0
        else:
            normal_px = 0.0

        total_px = dead_px + normal_px
        fraction = min(total_px / TOTAL_PIXELS, 1.0)

        pt_rows.append({
            "pointing_id":               int(pid),
            "pointing_ra":               round(meta["pointing_ra"],  5),
            "pointing_dec":              round(meta["pointing_dec"], 5),
            "pointing_filter":           meta["pointing_filter"],
            "pointing_exptime":          meta["pointing_exptime"],
            "pointing_night":            meta["pointing_night"],
            "rot_sky_pos":               round(meta["rot_sky_pos"], 4),
            "n_bright_streaks":          pdata["n_bright"],
            "n_normal_streaks":          pdata["n_normal"],
            "n_ccds_dead":               n_dead,
            "dead_ccd_ids":              str(sorted(dead)),
            "dead_pixel_loss":           dead_px,
            "normal_pixel_loss_clipped": round(normal_px, 2),
            "total_pixel_loss":          round(total_px, 2),
            "total_pixel_loss_fraction": round(fraction, 8),
        })

    f1_df = pd.DataFrame(dp_rows, columns=DP_COLS)
    f2_df = pd.DataFrame(pt_rows, columns=PT_COLS)
    f1_df.to_csv(f1_path, index=False)
    f2_df.to_csv(f2_path, index=False)

    elapsed = time.perf_counter() - t0
    n_bright_out = sum(1 for r in dp_rows if r["streak_category"] == "bright")
    n_normal_out = sum(1 for r in dp_rows if r["streak_category"] == "normal")
    log(f"[DONE]  {date_str} | {_hms(elapsed)} | "
        f"bright={n_bright_out:,} normal={n_normal_out:,} | "
        f"pointings={len(pt_rows):,}")

    del df, df_affected, df_valid
    gc.collect()

    return {
        "date":             date_str,
        "status":           "done",
        "elapsed":          elapsed,
        "n_pointings":      len(pt_rows),
        "n_bright_streaks": n_bright_out,
        "n_normal_streaks": n_normal_out,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Daily summary
# ─────────────────────────────────────────────────────────────────────────────

def append_daily_summary(output_dir, date_str):
    f2_path = os.path.join(output_dir, f"combined_pointings_{date_str}.csv")
    f3_path = os.path.join(output_dir, "combined_daily_summary.csv")
    if not os.path.isfile(f2_path):
        return
    try:
        f2 = pd.read_csv(f2_path)
    except pd.errors.EmptyDataError:
        return
    if f2.empty:
        return

    night = int(f2["pointing_night"].iloc[0]) if "pointing_night" in f2.columns else -1
    row = {
        "date":                           date_str,
        "pointing_night":                 night,
        "n_affected_pointings":           len(f2),
        "total_bright_streaks":           int(f2["n_bright_streaks"].sum()),
        "total_normal_streaks":           int(f2["n_normal_streaks"].sum()),
        "mean_ccds_dead_per_pointing":    round(f2["n_ccds_dead"].mean(), 4),
        "max_ccds_dead_per_pointing":     int(f2["n_ccds_dead"].max()),
        "mean_total_pixel_loss_fraction": round(f2["total_pixel_loss_fraction"].mean(), 8),
        "max_total_pixel_loss_fraction":  round(f2["total_pixel_loss_fraction"].max(), 8),
    }
    write_header = not os.path.isfile(f3_path)
    pd.DataFrame([row], columns=SUMMARY_COLS).to_csv(
        f3_path, mode="a", header=write_header, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(
    step2_dir    = "step2_output",
    step4_dir    = "step4_output_20deg",
    step6_dir    = "step6_output_perband",
    output_dir   = "step5_combined_loss_output_mag7",
    opsim_db     = None,   # path to opsim sqlite DB (for rotSkyPos)
    n_workers    = 40,
    log_filename = "step5_combined_loss_mag7.log",
):
    log_path   = setup_logging(output_dir, log_filename)
    wall_start = time.perf_counter()

    log("=" * 70)
    log(f"  STEP 5 COMBINED LOSS  (v4 — with RotSkyPos orientation)")
    log(f"  MAG_LIMIT = {MAG_LIMIT}  |  189 science CCDs  |  rotation: ON")
    log("=" * 70)
    log(f"  step2_dir  : {step2_dir}")
    log(f"  step4_dir  : {step4_dir}")
    log(f"  step6_dir  : {step6_dir}")
    log(f"  output_dir : {output_dir}")
    log(f"  opsim_db   : {opsim_db}")
    log(f"  n_workers  : {n_workers}")
    log("")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load rotSkyPos lookup from opsim DB ───────────────────────────────
    if opsim_db is None:
        try:
            from rubin_sim.data import get_baseline
            opsim_db = get_baseline()
            log(f"[INFO] opsim DB (auto): {opsim_db}")
        except Exception:
            log("[WARN] Could not find opsim DB — rotSkyPos defaulting to 0.0 for all pointings")
            opsim_db = None

    if opsim_db and os.path.isfile(opsim_db):
        log(f"[INFO] Loading rotSkyPos from opsim DB …")
        rot_lookup = load_rot_sky_pos(opsim_db)
        log(f"[INFO] {len(rot_lookup):,} pointing rotations loaded  "
            f"(range {min(rot_lookup.values()):.1f}° – {max(rot_lookup.values()):.1f}°)")
    else:
        log("[WARN] opsim DB not available — rotSkyPos = 0.0 for all (no rotation)")
        rot_lookup = {}

    # ── Find step2 files ──────────────────────────────────────────────────
    all_step2 = sorted(glob.glob(
        os.path.join(step2_dir, "streak_trajectories_*.csv")))
    all_step2 = [p for p in all_step2
                 if "_azel" not in p and "_bright" not in p and "_raft" not in p]

    if not all_step2:
        log(f"[WARN] No step2 CSVs found in {step2_dir}")
        return

    log(f"[INFO] {len(all_step2)} day CSV(s) found")

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

    tasks = []
    n_skip = 0
    for step2_path in all_step2:
        date_str = (os.path.basename(step2_path)
                    .replace("streak_trajectories_", "").replace(".csv", ""))
        bright_path = os.path.join(
            step4_dir, f"streak_trajectories_{date_str}_bright_corrected.csv")
        f1 = os.path.join(output_dir, f"combined_datapoints_{date_str}.csv")
        f2 = os.path.join(output_dir, f"combined_pointings_{date_str}.csv")
        if os.path.isfile(f1) and os.path.isfile(f2):
            n_skip += 1
            continue
        # Pass rot_lookup to each worker
        tasks.append((step2_path, bright_path, step6_dir,
                      output_dir, log_path, rot_lookup))

    log(f"[INFO] {len(tasks)} days to process  ({n_skip} already done)\n")

    if not tasks:
        log("[INFO] Nothing to do.")
        return

    with Pool(processes=min(n_workers, len(tasks))) as pool:
        results = pool.map(process_day, tasks)

    log("\n[INFO] Building daily summary …")
    for task in tasks:
        step2_path = task[0]
        date_str = (os.path.basename(step2_path)
                    .replace("streak_trajectories_", "").replace(".csv", ""))
        append_daily_summary(output_dir, date_str)

    log("\n" + "=" * 70)
    n_done   = sum(1 for r in results if r["status"] == "done")
    n_errors = sum(1 for r in results if r["status"] == "error")
    n_empty  = sum(1 for r in results if "empty"     in r.get("status","")
                                      or "no_bright" in r.get("status",""))
    log(f"  Done        : {n_done}")
    log(f"  No bright   : {n_empty}")
    log(f"  Errors      : {n_errors}")

    if n_done > 0:
        done = [r for r in results if r["status"] == "done"]
        log(f"  Avg time/day        : {_hms(sum(r['elapsed'] for r in done)/n_done)}")
        log(f"  Total bright streaks: {sum(r['n_bright_streaks'] for r in done):,}")
        log(f"  Total normal streaks: {sum(r['n_normal_streaks'] for r in done):,}")
        log(f"  Total pointings     : {sum(r['n_pointings'] for r in done):,}")

    f3 = os.path.join(output_dir, "combined_daily_summary.csv")
    if os.path.isfile(f3):
        summary = pd.read_csv(f3)
        log(f"\n  Daily summary ({len(summary)} days):")
        log(f"  Mean total loss fraction : "
            f"{summary['mean_total_pixel_loss_fraction'].mean():.6f}")
        log(f"  Max  total loss fraction : "
            f"{summary['max_total_pixel_loss_fraction'].max():.6f}")
        log(f"  Max  CCDs dead/pointing  : "
            f"{summary['max_ccds_dead_per_pointing'].max()}")

    log(f"\n  Total wall time: {_hms(time.perf_counter() - wall_start)}")
    log("[DONE]")


if __name__ == "__main__":
    main(
        step2_dir  = "step2_output",
        step4_dir  = "step4b_output_rad",
        step6_dir  = "step6_output_perband_rad",
        output_dir = "step5_combined_loss_output_rad",
        opsim_db   = "baseline_v5.3.0_10yrs.db",   # auto-detected via rubin_sim.data.get_baseline()
        n_workers  = 94,
    )