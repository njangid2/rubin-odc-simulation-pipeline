#!/usr/bin/env python3
"""
step9_pixel_loss.py
─────────────────────────────────────────────────────────────────────────────
Runs the v2 pixel-level saturation/overlap physics (originally
pixel_level_pointing.py, written for ONE hand-picked date/pointing_id) over
EVERY pointing in the WHOLE simulation, as "step9" of the pipeline.

This is a single self-contained file — no separate lib/ import required,
so it can be dropped anywhere (including a SLURM working directory) without
worrying about relative import paths.

PHYSICS (unchanged from the original pixel_level_pointing.py v2)
─────────────────────────────────────────────────────────────────
    - band-corrected ab_magnitude (step4b's ab_magnitude_corrected) decides
      BRIGHT vs NORMAL streaks (MAG_LIMIT = 2.0 mag).
    - BRIGHT streaks: every CCD touched is entirely dead (no per-pixel
      electron accounting — the CCD is already 100% lost).
    - NORMAL streaks: per-band SURFACE BRIGHTNESS from step6 (not the raw
      mag) gives peak_electrons, using the real camera rotation (RotSkyPos).
    - Each normal streak gets a mask polygon (100px half-width if
      unsaturated, 500px if saturated) along its actual path. Where two
      streaks' mask polygons physically overlap, their electrons are
      summed for THAT overlap region only, and if the combined electrons
      cross FULL_WELL_E (130,000 e-), only that sub-area is upgraded to
      "saturated" — not the whole streak.
    - OLD total pixel loss = dead CCDs + union of each streak's own
      (individual-status) mask polygon — i.e. exactly step8's number, with
      no overlap awareness.
    - NEW total pixel loss = dead CCDs + union of each streak's mask
      polygon rebuilt with the wider width ONLY along the segments that
      actually fall inside a real overlap region.

WHAT THIS FILE ADDS ON TOP (the "step9" / whole-sim piece)
───────────────────────────────────────────────────────────
    - Discovers every date's streak_trajectories_YYYY-MM-DD.csv under
      --step2-dir.
    - For each date, loads step2 + step4b ONCE (not once per pointing),
      loads all six step6 per-band CSVs ONCE, and batch-fetches every
      pointing's rotSkyPos in a SINGLE sqlite query — instead of the
      original per-pointing load_pointing(), which would mean thousands of
      redundant full-file reads across a whole sim.
    - Loops every pointing_id in that date through the identical physics
      above and appends one summary row per pointing to a CSV.
    - Resumable: rerunning with the same --out skips (date, pointing_id)
      rows already written.
    - A single bad pointing or a corrupt date is logged as an error row
      and does NOT crash the rest of the run.
    - Parallelizes across DATES (--workers), since each date's expensive
      I/O is already batched once per date.

INPUTS (same layout as the original script)
────────────────────────────────────────────
    step2_output/streak_trajectories_YYYY-MM-DD.csv
    step4b_output_rad/streak_trajectories_YYYY-MM-DD_bright_corrected.csv
    step6_output_perband_rad/step6_YYYY-MM-DD_{band}.csv
    opsim DB (baseline_v5.3.0_10yrs.db) for rotSkyPos

USAGE
─────
    # everything below has a matching hardcoded default, so this alone
    # runs the whole sim with 90 parallel workers:
    python3 step9_pixel_loss.py

    # override anything if needed:
    python3 step9_pixel_loss.py --max-dates 2 --workers 1 --verbose

    # spot-check plots (3-panel NAIVE/COMBINED/MASK-POLYGON diagram) for
    # the N pointings with the biggest OLD->NEW delta:
    python3 step9_pixel_loss.py --plot-top-n 20 --plot-dir step9_output/plots
"""

import argparse
import csv
import gc
import glob
import math
import os
import re
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe (SLURM/cluster nodes have no display)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch, Polygon as MplPolygon, Circle
from matplotlib.lines import Line2D
from shapely.geometry import Polygon
from shapely.ops import unary_union


# ═════════════════════════════════════════════════════════════════════════
# PART 1 — Physics constants (identical to pixel_level_pointing.py / step8)
# ═════════════════════════════════════════════════════════════════════════

MAG_LIMIT = 2.0   # streaks with ab_mag_corrected < MAG_LIMIT -> whole CCD dead

CCD_SIZE_MM   = 42.0
GAP_CCD_MM    = 0.27
GAP_RAFT_MM   = 0.50
RAFT_SIZE_MM  = 3 * CCD_SIZE_MM + 2 * GAP_CCD_MM
STEP_MM       = RAFT_SIZE_MM + GAP_RAFT_MM
TOTAL_SPAN_MM = 5 * STEP_MM - GAP_RAFT_MM
FOV_RADIUS_MM = TOTAL_SPAN_MM / 2
FOV_DEG       = 1.75
MM_PER_DEG    = FOV_RADIUS_MM / FOV_DEG
DEG_PER_MM    = FOV_DEG / FOV_RADIUS_MM

PLATE_SCALE_ARCSEC_PX = 0.2
PX_PER_DEG    = 3600.0 / PLATE_SCALE_ARCSEC_PX
PX_PER_MM     = PX_PER_DEG * DEG_PER_MM

CCD_SIZE_PX_PHYS = CCD_SIZE_MM * PX_PER_MM
CCD_PX        = 4096
CCD_PIXELS    = CCD_PX * CCD_PX
TOTAL_PIXELS  = 3.2e9
FOV_RADIUS_DEG = FOV_DEG
FOV_RADIUS_PX  = FOV_RADIUS_DEG * 3600.0 / PLATE_SCALE_ARCSEC_PX   # ~31,500 px

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

STATUS_ORDER = {"faint": -1, "unsaturated": 0, "saturated": 1, "ultrabright": 2}

STATUS_COLORS = {
    "unsaturated": "#4C72B0", "saturated": "#DD8452",
    "ultrabright": "#C44E52", "faint": "#AAAAAA",
}
PLOT_STATUSES = ["unsaturated", "saturated", "ultrabright"]
CMAP = ListedColormap([STATUS_COLORS[s] for s in PLOT_STATUSES])

# ─────────────────────────────────────────────────────────────────────────
# CCD grid — identical construction to step8_ccd_pixel.py
# ─────────────────────────────────────────────────────────────────────────

GRID_TYPE = [
    [ 'WF', 'ITL', 'ITL', 'ITL',  'WF'],
    ['e2V', 'e2V', 'e2V', 'e2V', 'e2V'],
    ['ITL', 'e2V', 'e2V', 'e2V', 'e2V'],
    ['ITL', 'e2V', 'e2V', 'e2V', 'e2V'],
    [ 'WF', 'ITL', 'ITL', 'ITL',  'WF'],
]

CCD_BBOX_PX = {}
_ccd_n = 1
for _rr in range(5):
    for _rc in range(5):
        if GRID_TYPE[_rr][_rc] not in ('ITL', 'e2V'):
            continue
        _raft_x_mm = -FOV_RADIUS_MM + _rc * STEP_MM
        _raft_y_mm =  FOV_RADIUS_MM - _rr * STEP_MM - RAFT_SIZE_MM
        for _cr in range(3):
            for _cc in range(3):
                _ccd_x_mm = _raft_x_mm + _cc * (CCD_SIZE_MM + GAP_CCD_MM)
                _ccd_y_mm = _raft_y_mm + (2 - _cr) * (CCD_SIZE_MM + GAP_CCD_MM)
                _x0 = _ccd_x_mm * PX_PER_MM
                _y0 = _ccd_y_mm * PX_PER_MM
                _x1 = _x0 + CCD_SIZE_PX_PHYS
                _y1 = _y0 + CCD_SIZE_PX_PHYS
                CCD_BBOX_PX[_ccd_n] = (_x0, _x1, _y0, _y1)
                _ccd_n += 1

_CCD_IDS   = np.array(list(CCD_BBOX_PX.keys()), dtype=np.int16)
_CCD_BOXES = np.array(list(CCD_BBOX_PX.values()), dtype=np.float64)

# ── FIX: single source of truth for "real CCD footprint" + consistent
# TOTAL_PIXELS denominator. Both close out two related bugs found in
# production:
#   1. old_normal_area/new_normal_area (mask-polygon area) was never
#      clipped to real CCD boundaries, so any streak crossing a WF-corner
#      gap or inter-CCD/inter-raft gap counted phantom "lost" area that
#      has no physical CCD behind it at all.
#   2. dead_pixel_loss (whole dead CCDs) and old/new_normal_area were
#      simply ADDED together with no check for overlap between them --
#      a normal streak's mask lying inside an already-100%-dead CCD
#      added its own area AGAIN on top of that already-counted CCD,
#      which is how old_total_fraction/new_total_fraction were observed
#      exceeding 100% for pointings with many streaks and/or many dead
#      CCDs.
# _CCD_FOOTPRINT_UNION is the union of every real CCD box (built from
# this exact CCD_BBOX_PX geometry, so it's self-consistent with whatever
# convention CCD_SIZE_PX_PHYS uses -- no mixing with the separate nominal
# CCD_PX=4096 constant). TOTAL_PIXELS is reassigned here (after real
# geometry exists) to this union's exact area, replacing the original
# hardcoded 3.2e9 approximation, so numerator and denominator always use
# the same definition of "one real pixel."
from shapely.geometry import box as _shapely_box
_CCD_FOOTPRINT_UNION = unary_union(
    [_shapely_box(x0, y0, x1, y1) for (x0, x1, y0, y1) in CCD_BBOX_PX.values()]
)
TOTAL_PIXELS = _CCD_FOOTPRINT_UNION.area


# ═════════════════════════════════════════════════════════════════════════
# PART 2 — Helpers (identical to pixel_level_pointing.py / step8)
# ═════════════════════════════════════════════════════════════════════════

def base_band(filter_value) -> str:
    return str(filter_value).strip().split('_')[0]


def radec_to_focal_px(ra_deg, dec_deg, ra0_deg, dec0_deg, rot_sky_pos_deg=0.0):
    ra, dec   = np.radians(ra_deg), np.radians(dec_deg)
    ra0, dec0 = np.radians(ra0_deg), np.radians(dec0_deg)
    cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    x_rad = np.cos(dec) * np.sin(ra - ra0) / cos_c
    y_rad = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_c
    x_sky = np.degrees(x_rad) * PX_PER_DEG
    y_sky = np.degrees(y_rad) * PX_PER_DEG

    angle_rad = np.radians(180.0 - rot_sky_pos_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    x_cam =  cos_a * x_sky + sin_a * y_sky
    y_cam = -sin_a * x_sky + cos_a * y_sky
    return x_cam, y_cam


def angular_sep_deg(ra1, dec1, ra2, dec2):
    r1, d1 = np.radians(ra1), np.radians(dec1)
    r2, d2 = np.radians(ra2), np.radians(dec2)
    cos_a = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def interp_fov_boundary(ra_in, dec_in, ra_out, dec_out, pt_ra, pt_dec):
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid   = 0.5 * (lo + hi)
        ra_m  = ra_in  + mid * (ra_out  - ra_in)
        dec_m = dec_in + mid * (dec_out - dec_in)
        if angular_sep_deg(pt_ra, pt_dec, ra_m, dec_m) < FOV_RADIUS_DEG:
            lo = mid
        else:
            hi = mid
    f = 0.5 * (lo + hi)
    return ra_in + f * (ra_out - ra_in), dec_in + f * (dec_out - dec_in)


def streak_waypoints_px(grp, pt_ra, pt_dec, rot_sky_pos_deg=0.0):
    fov    = grp['in_fov'].values
    in_idx = np.where(fov)[0]
    if len(in_idx) == 0:
        return None, None

    ra_arr, dec_arr = grp['ra_deg'].values, grp['dec_deg'].values
    first_in, last_in = in_idx[0], in_idx[-1]
    entering = first_in > 0
    exiting  = last_in < len(grp) - 1

    wp_ra, wp_dec = [], []
    if entering:
        r_bnd, d_bnd = interp_fov_boundary(
            ra_arr[first_in], dec_arr[first_in],
            ra_arr[first_in - 1], dec_arr[first_in - 1],
            pt_ra, pt_dec)
        wp_ra.append(r_bnd); wp_dec.append(d_bnd)
    for i in in_idx:
        wp_ra.append(ra_arr[i]); wp_dec.append(dec_arr[i])
    if exiting:
        r_bnd, d_bnd = interp_fov_boundary(
            ra_arr[last_in], dec_arr[last_in],
            ra_arr[last_in + 1], dec_arr[last_in + 1],
            pt_ra, pt_dec)
        wp_ra.append(r_bnd); wp_dec.append(d_bnd)

    wp_x, wp_y = radec_to_focal_px(
        np.array(wp_ra), np.array(wp_dec), pt_ra, pt_dec,
        rot_sky_pos_deg=rot_sky_pos_deg
    )
    return wp_x, wp_y


def point_to_ccds(x_px, y_px):
    mask = (
        (x_px >= _CCD_BOXES[:, 0]) & (x_px <= _CCD_BOXES[:, 1]) &
        (y_px >= _CCD_BOXES[:, 2]) & (y_px <= _CCD_BOXES[:, 3])
    )
    return set(_CCD_IDS[mask].tolist())


def segment_to_ccds(x0, y0, x1, y1, n_interp=50):
    ts = np.linspace(0.0, 1.0, n_interp + 2)
    xs, ys = x0 + ts * (x1 - x0), y0 + ts * (y1 - y0)
    hit = set()
    for x, y in zip(xs, ys):
        hit |= point_to_ccds(x, y)
    return hit


def streak_to_ccds(wp_x, wp_y):
    hit = set()
    for k in range(len(wp_x)):
        hit |= point_to_ccds(wp_x[k], wp_y[k])
    for k in range(len(wp_x) - 1):
        hit |= segment_to_ccds(wp_x[k], wp_y[k], wp_x[k + 1], wp_y[k + 1])
    return hit


def sb_to_peak_electrons(sb, band, t_crossing_sec):
    if np.isnan(sb) or np.isnan(t_crossing_sec) or t_crossing_sec <= 0:
        return np.nan
    if band not in BAND_ZP:
        return np.nan
    zp = BAND_ZP[band]
    flux_per_arcsec2 = 10.0 ** ((zp - sb) / 2.5)
    e_per_px_per_sec = flux_per_arcsec2 * PIXEL_AREA_ARCSEC2
    return e_per_px_per_sec * t_crossing_sec * PSF_PEAK_FRAC


def classify_status_full(sb, band, t_crossing_sec):
    if np.isnan(sb):
        return "faint", np.nan
    peak_e = sb_to_peak_electrons(sb, band, t_crossing_sec)
    if np.isnan(peak_e):
        return "faint", np.nan
    status = "saturated" if peak_e > FULL_WELL_E else "unsaturated"
    return status, peak_e


def classify_from_electrons(peak_e):
    if np.isnan(peak_e):
        return "faint"
    return "saturated" if peak_e > FULL_WELL_E else "unsaturated"


def rasterize_segment(x0, y0, x1, y1, step_px=1.0):
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0:
        return [(int(round(x0)), int(round(y0)))]
    n = max(int(length / step_px), 1)
    xs = np.linspace(x0, x1, n + 1)
    ys = np.linspace(y0, y1, n + 1)
    return list(zip(np.round(xs).astype(int), np.round(ys).astype(int)))


# ═════════════════════════════════════════════════════════════════════════
# PART 3 — rotSkyPos lookup (per-pointing, and batched-per-date for step9)
# ═════════════════════════════════════════════════════════════════════════

def get_rot_sky_pos(pointing_id, opsim_db=None, override=None):
    if override is not None:
        return override
    db_path = opsim_db
    if db_path is None or not os.path.isfile(db_path):
        return 0.0
    try:
        with sqlite3.connect(db_path) as con:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'", con
            )["name"].tolist()
            table = next((t for t in ["observations", "SummaryAllProps", "Summary"]
                          if t in tables), None)
            row = pd.read_sql(
                f"SELECT rotSkyPos FROM {table} WHERE observationId = {int(pointing_id)}",
                con
            )
        if len(row) == 0:
            return 0.0
        return float(row["rotSkyPos"].iloc[0])
    except Exception:
        return 0.0


def get_rot_sky_pos_batch(pointing_ids, opsim_db=None, override=None):
    """One SQL query for ALL pointing_ids in a date, instead of one query
    per pointing. Falls back to 0.0 for any pointing not found / on any
    failure — same fallback behavior as get_rot_sky_pos()."""
    pointing_ids = list(pointing_ids)
    if override is not None:
        return {pid: override for pid in pointing_ids}

    result = {pid: 0.0 for pid in pointing_ids}
    if opsim_db is None or not os.path.isfile(opsim_db):
        return result

    try:
        with sqlite3.connect(opsim_db) as con:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'", con
            )["name"].tolist()
            table = next((t for t in ["observations", "SummaryAllProps", "Summary"]
                          if t in tables), None)
            if table is None:
                return result
            ids_str = ",".join(str(int(p)) for p in pointing_ids)
            df = pd.read_sql(
                f"SELECT observationId, rotSkyPos FROM {table} "
                f"WHERE observationId IN ({ids_str})",
                con
            )
        for _, row in df.iterrows():
            result[int(row["observationId"])] = float(row["rotSkyPos"])
    except Exception:
        pass
    return result


# ═════════════════════════════════════════════════════════════════════════
# PART 4 — Loading data: per-date (once) then per-pointing (sliced)
# ═════════════════════════════════════════════════════════════════════════

def load_date_frames(date_str, step2_dir, step4b_dir):
    """Reads step2 + step4b ONCE for the whole date (all pointings).

    MEMORY NOTE: dtype is pinned for the low-cardinality/integer columns
    (category for repeated strings, smaller int width for IDs/counters).
    This does NOT change any physics or numeric precision -- ra_deg/dec_deg/
    ab_magnitude/exptime stay float64 exactly as before. It only stops
    pandas from storing millions of repeated Python string objects for
    sat_name/pointing_filter, which was the dominant memory cost on large
    (150-250MB) step2 files and is what caused OOM kills at high --workers.
    """
    step2_path  = os.path.join(step2_dir, f"streak_trajectories_{date_str}.csv")
    step4b_path = os.path.join(
        step4b_dir, f"streak_trajectories_{date_str}_bright_corrected.csv"
    )

    df = pd.read_csv(
        step2_path,
        usecols=[
            "pointing_id", "pointing_ra", "pointing_dec",
            "pointing_filter", "pointing_exptime", "pointing_night",
            "shell_id", "sat_name", "step",
            "ra_deg", "dec_deg", "in_fov", "sunlit",
        ],
        dtype={
            "pointing_id": "int32",
            "pointing_night": "int32",
            "pointing_filter": "category",
            "sat_name": "category",
            "shell_id": "int16",
            "step": "int16",
            "in_fov": "bool",
            "sunlit": "bool",
        },
    )
    corrected = pd.read_csv(step4b_path, usecols=["ab_magnitude_corrected"])

    if len(df) != len(corrected):
        raise ValueError(
            f"Row mismatch for {date_str}: step2={len(df)} step4b={len(corrected)}"
        )
    df["ab_magnitude"] = corrected["ab_magnitude_corrected"].values
    del corrected
    return df


def load_step6_for_date(date_str, step6_dir):
    """Reads all six per-band step6 CSVs for a date ONCE, builds an O(1)
    (pointing_id, sat_name, shell_id) -> {sb, t_crossing_sec} lookup.
    Same dtype-pinning rationale as load_date_frames (see its docstring)."""
    sb_lookup = {}
    for band in "ugrizy":
        s6_path = os.path.join(step6_dir, f"step6_{date_str}_{band}.csv")
        if not os.path.isfile(s6_path):
            continue
        try:
            s6 = pd.read_csv(
                s6_path,
                usecols=[
                    "pointing_id", "sat_name", "shell_id",
                    "surface_brightness_mag_arcsec2", "t_crossing_sec",
                ],
                dtype={
                    "pointing_id": "int32",
                    "sat_name": "category",
                    "shell_id": "int16",
                },
            )
        except Exception:
            continue
        for row in s6.itertuples(index=False):
            key = (int(row.pointing_id), row.sat_name, int(row.shell_id))
            sb_lookup[key] = {
                "sb": row.surface_brightness_mag_arcsec2,
                "t_crossing_sec": row.t_crossing_sec,
            }
        del s6
    return sb_lookup


def build_pointing_from_frames(df_date, pointing_id, sb_lookup_date, rot_sky_pos,
                                date_str=None):
    """Same per-pointing streak-building logic as the original
    load_pointing(), but sliced from already-loaded date-level data."""
    df = df_date[df_date["pointing_id"] == pointing_id].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No rows for pointing_id={pointing_id}")

    pt_ra  = float(df["pointing_ra"].iloc[0])
    pt_dec = float(df["pointing_dec"].iloc[0])
    t_exp  = float(df["pointing_exptime"].iloc[0])
    night  = int(df["pointing_night"].iloc[0])
    filt   = base_band(df["pointing_filter"].iloc[0])

    valid_mask = df["in_fov"] & df["sunlit"] & df["ab_magnitude"].notna()
    df_valid = df[valid_mask]
    if df_valid.empty:
        min_ab = pd.DataFrame(columns=["sat_name", "shell_id", "min_ab_mag"])
    else:
        min_ab = (df_valid.groupby(["sat_name", "shell_id"], observed=True)["ab_magnitude"]
                  .min().reset_index()
                  .rename(columns={"ab_magnitude": "min_ab_mag"}))

    bright_streaks = []
    normal_streaks = []

    for (sat_name, shell_id), grp in df.groupby(["sat_name", "shell_id"], observed=True):
        grp = grp.sort_values("step").reset_index(drop=True)

        m = min_ab[(min_ab["sat_name"] == sat_name) &
                    (min_ab["shell_id"] == shell_id)]
        min_ab_val = float(m["min_ab_mag"].iloc[0]) if len(m) else np.nan
        if np.isnan(min_ab_val):
            continue

        wp_x, wp_y = streak_waypoints_px(grp, pt_ra, pt_dec, rot_sky_pos)
        if wp_x is None:
            continue

        hit_ccds = streak_to_ccds(wp_x, wp_y)

        if min_ab_val < MAG_LIMIT:
            bright_streaks.append({
                "sat_name": sat_name, "shell_id": shell_id,
                "min_ab_magnitude": min_ab_val, "hit_ccds": hit_ccds,
                "wp_x": wp_x, "wp_y": wp_y,
                "individual_status": "ultrabright",
            })
        else:
            key = (int(pointing_id), sat_name, int(shell_id))
            s6 = sb_lookup_date.get(key, {})
            sb = s6.get("sb", np.nan)
            t_crossing_sec = s6.get("t_crossing_sec", np.nan)

            status, peak_e = classify_status_full(sb, filt, t_crossing_sec)

            path_px = []
            if len(wp_x) >= 2:
                for i in range(len(wp_x) - 1):
                    path_px.extend(rasterize_segment(
                        wp_x[i], wp_y[i], wp_x[i + 1], wp_y[i + 1]
                    ))
            else:
                path_px = [(int(round(wp_x[0])), int(round(wp_y[0])))]
            path_px = list(dict.fromkeys(path_px))

            normal_streaks.append({
                "sat_name": sat_name, "shell_id": shell_id,
                "min_ab_magnitude": min_ab_val,
                "surface_brightness": sb, "t_crossing_sec": t_crossing_sec,
                "peak_electrons": peak_e,
                "individual_status": status,
                "hit_ccds": hit_ccds,
                "pixels": path_px,
                "wp_x": wp_x, "wp_y": wp_y,
            })

    meta = {
        "pointing_id": pointing_id, "date": date_str,
        "pointing_ra": pt_ra, "pointing_dec": pt_dec,
        "pointing_exptime": t_exp, "pointing_night": night,
        "pointing_filter": filt, "rot_sky_pos": rot_sky_pos,
        "n_bright_streaks": len(bright_streaks),
        "n_normal_streaks": len(normal_streaks),
    }
    return bright_streaks, normal_streaks, meta


def load_pointing(date_str, pointing_id, step2_dir, step4b_dir, step6_dir,
                   opsim_db=None, rot_sky_pos_override=None):
    """Original single-pointing entry point (reads everything from disk for
    just this one pointing). Handy for ad-hoc debugging of one pointing;
    the whole-sim run below uses the per-date-cached path instead."""
    df_date = load_date_frames(date_str, step2_dir, step4b_dir)
    if not (df_date["pointing_id"] == pointing_id).any():
        raise ValueError(f"No rows for pointing_id={pointing_id} on {date_str}")
    rot_sky_pos = get_rot_sky_pos(pointing_id, opsim_db, rot_sky_pos_override)
    sb_lookup = load_step6_for_date(date_str, step6_dir)
    return build_pointing_from_frames(df_date, pointing_id, sb_lookup, rot_sky_pos,
                                       date_str=date_str)


# ═════════════════════════════════════════════════════════════════════════
# PART 5 — Pixel/overlap accounting (identical to pixel_level_pointing.py)
# ═════════════════════════════════════════════════════════════════════════

def accumulate_pixels(normal_streaks):
    pixel_combined_e = {}
    pixel_contributors = {}
    for si, s in enumerate(normal_streaks):
        if np.isnan(s["peak_electrons"]):
            continue
        for px in s["pixels"]:
            pixel_combined_e[px] = pixel_combined_e.get(px, 0.0) + s["peak_electrons"]
            pixel_contributors.setdefault(px, []).append(si)
    return pixel_combined_e, pixel_contributors


def evaluate_upgrades(normal_streaks, pixel_combined_e):
    for s in normal_streaks:
        has_data = s["individual_status"] != "faint"
        worst_combined = s["individual_status"] if has_data else "unsaturated"
        touched_by_others = False
        for px in s["pixels"]:
            e = pixel_combined_e.get(px, 0.0)
            st = classify_from_electrons(e)
            if STATUS_ORDER.get(st, -1) > STATUS_ORDER.get(worst_combined, -1):
                worst_combined = st
            if not has_data and e > 0:
                touched_by_others = True
        s["combined_status"] = worst_combined if has_data else s["individual_status"]
        s["upgraded"] = (
            has_data and
            STATUS_ORDER.get(worst_combined, -1) > STATUS_ORDER.get(s["individual_status"], -1)
        )
        s["affected_despite_no_data"] = (not has_data) and touched_by_others and (
            STATUS_ORDER.get(worst_combined, -1) > STATUS_ORDER["unsaturated"]
        )
    return normal_streaks


def make_streak_polygon(wp_x, wp_y, half_width_px):
    polys = []
    for k in range(len(wp_x) - 1):
        x0, y0 = wp_x[k], wp_y[k]
        x1, y1 = wp_x[k + 1], wp_y[k + 1]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        nx, ny = -dy / length, dx / length
        corners = [
            (x0 + nx * half_width_px, y0 + ny * half_width_px),
            (x1 + nx * half_width_px, y1 + ny * half_width_px),
            (x1 - nx * half_width_px, y1 - ny * half_width_px),
            (x0 - nx * half_width_px, y0 - ny * half_width_px),
        ]
        p = Polygon(corners)
        if p.is_valid and not p.is_empty:
            polys.append(p)
    if not polys:
        return None
    result = unary_union(polys)
    return result if not result.is_empty else None


def build_mask_polygons(normal_streaks):
    for s in normal_streaks:
        s["mask_polygon"] = None
        if s["individual_status"] == "faint":
            continue
        W = MASK_WIDTH_PX.get(s["individual_status"], 0)
        if W <= 0:
            continue
        poly = make_streak_polygon(s["wp_x"], s["wp_y"], W / 2.0)
        s["mask_polygon"] = poly
    return normal_streaks


def find_polygon_overlaps(normal_streaks):
    n = len(normal_streaks)
    for s in normal_streaks:
        s["combined_status_mask"] = s["individual_status"]
        s["upgraded_mask"] = False
        s["_upgrade_geoms"] = []
        s["own_area_px2"] = s["mask_polygon"].area if s["mask_polygon"] is not None else 0.0

    overlaps = []
    for i in range(n):
        si = normal_streaks[i]
        if si["mask_polygon"] is None or np.isnan(si["peak_electrons"]):
            continue
        for j in range(i + 1, n):
            sj = normal_streaks[j]
            if sj["mask_polygon"] is None or np.isnan(sj["peak_electrons"]):
                continue
            if not si["mask_polygon"].intersects(sj["mask_polygon"]):
                continue
            inter = si["mask_polygon"].intersection(sj["mask_polygon"])
            if inter.is_empty or inter.area <= 0:
                continue

            combined_e = si["peak_electrons"] + sj["peak_electrons"]
            combined_status = classify_from_electrons(combined_e)

            overlaps.append({
                "i": i, "j": j,
                "sat_i": si["sat_name"], "sat_j": sj["sat_name"],
                "area_px2": inter.area,
                "combined_e": combined_e,
                "combined_status": combined_status,
                "intersection": inter,
            })

            for s in (si, sj):
                if STATUS_ORDER.get(combined_status, -1) > STATUS_ORDER.get(s["combined_status_mask"], -1):
                    s["combined_status_mask"] = combined_status
                if STATUS_ORDER.get(combined_status, -1) > STATUS_ORDER.get(s["individual_status"], -1):
                    s["_upgrade_geoms"].append(inter)

    for s in normal_streaks:
        geoms = s.pop("_upgrade_geoms")
        if geoms:
            merged = unary_union(geoms)
            s["upgraded_geom"] = merged
            s["upgraded_area_px2"] = merged.area
            s["upgraded_mask"] = True
        else:
            s["upgraded_geom"] = None
            s["upgraded_area_px2"] = 0.0
        s["upgraded_fraction"] = (
            s["upgraded_area_px2"] / s["own_area_px2"] if s["own_area_px2"] > 0 else 0.0
        )

    return overlaps


def make_streak_polygon_variable_width(wp_x, wp_y, half_widths):
    polys = []
    for k in range(len(wp_x) - 1):
        x0, y0 = wp_x[k], wp_y[k]
        x1, y1 = wp_x[k + 1], wp_y[k + 1]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        nx, ny = -dy / length, dx / length
        hw = half_widths[k]
        corners = [
            (x0 + nx * hw, y0 + ny * hw),
            (x1 + nx * hw, y1 + ny * hw),
            (x1 - nx * hw, y1 - ny * hw),
            (x0 - nx * hw, y0 - ny * hw),
        ]
        p = Polygon(corners)
        if p.is_valid and not p.is_empty:
            polys.append(p)
    if not polys:
        return None
    result = unary_union(polys)
    return result if not result.is_empty else None


def rebuild_polygons_precise(normal_streaks):
    for s in normal_streaks:
        s["mask_polygon_new"] = None
        wp_x, wp_y = s.get("wp_x"), s.get("wp_y")
        if wp_x is None or len(wp_x) < 2:
            continue

        base_status = s["individual_status"]
        if base_status == "faint":
            continue
        base_W = MASK_WIDTH_PX.get(base_status, 0)
        if base_W <= 0:
            continue

        worse_status = s.get("combined_status_mask", base_status)
        worse_W = MASK_WIDTH_PX.get(worse_status, base_W)
        upgraded_geom = s.get("upgraded_geom")

        n_seg = len(wp_x) - 1
        half_widths = np.full(n_seg, base_W / 2.0)

        if upgraded_geom is not None and not upgraded_geom.is_empty and worse_W > base_W:
            for k in range(n_seg):
                x0, y0 = wp_x[k], wp_y[k]
                x1, y1 = wp_x[k + 1], wp_y[k + 1]
                dx, dy = x1 - x0, y1 - y0
                length = math.hypot(dx, dy)
                if length == 0:
                    continue
                nx, ny = -dy / length, dx / length
                hw = base_W / 2.0
                corners = [
                    (x0 + nx * hw, y0 + ny * hw),
                    (x1 + nx * hw, y1 + ny * hw),
                    (x1 - nx * hw, y1 - ny * hw),
                    (x0 - nx * hw, y0 - ny * hw),
                ]
                seg_poly = Polygon(corners)
                if not seg_poly.is_valid or seg_poly.is_empty:
                    continue
                if seg_poly.intersects(upgraded_geom):
                    if seg_poly.intersection(upgraded_geom).area > 0:
                        half_widths[k] = worse_W / 2.0

        s["mask_polygon_new"] = make_streak_polygon_variable_width(wp_x, wp_y, half_widths)
    return normal_streaks


def compute_pixel_loss_totals(bright_streaks, normal_streaks):
    """
    FIXED: dead-CCD footprints and normal-streak masks are combined into
    ONE union before measuring area (instead of being added as two
    separately-computed numbers), and clipped to the real CCD footprint
    (_CCD_FOOTPRINT_UNION). This closes two bugs:
      - double-counting when a normal streak's mask overlaps a CCD that's
        already entirely dead from a bright streak
      - phantom area from masks crossing WF-corner gaps / inter-CCD gaps,
        which don't correspond to any real CCD pixels at all
    Both together were the cause of old_total_fraction/new_total_fraction
    occasionally exceeding 100% for pointings with many streaks and/or
    many dead CCDs -- a physically impossible value for a genuine
    pixel-count fraction, now bounded correctly at <=100%.
    """
    dead_ccds = set()
    for b in bright_streaks:
        dead_ccds |= b["hit_ccds"]
    dead_ccd_polys = [
        Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        for c in dead_ccds for (x0, x1, y0, y1) in [CCD_BBOX_PX[c]]
    ]
    dead_pixel_loss = unary_union(dead_ccd_polys).area if dead_ccd_polys else 0.0

    old_polys = [s["mask_polygon"] for s in normal_streaks if s.get("mask_polygon") is not None]
    new_polys = [s["mask_polygon_new"] for s in normal_streaks if s.get("mask_polygon_new") is not None]

    old_combined = unary_union(dead_ccd_polys + old_polys) if (dead_ccd_polys or old_polys) else None
    new_combined = unary_union(dead_ccd_polys + new_polys) if (dead_ccd_polys or new_polys) else None

    old_total = old_combined.intersection(_CCD_FOOTPRINT_UNION).area if old_combined is not None else 0.0
    new_total = new_combined.intersection(_CCD_FOOTPRINT_UNION).area if new_combined is not None else 0.0

    # kept for reporting/backward-compat -- the union-based old_total/
    # new_total above (not this sum) are what should be trusted now
    old_normal_area = unary_union(old_polys).area if old_polys else 0.0
    new_normal_area = unary_union(new_polys).area if new_polys else 0.0

    return {
        "dead_pixel_loss": dead_pixel_loss,
        "n_dead_ccds": len(dead_ccds),
        "old_normal_area_px2": old_normal_area,
        "new_normal_area_px2": new_normal_area,
        "old_total_px2": old_total,
        "new_total_px2": new_total,
        "old_total_fraction": old_total / TOTAL_PIXELS,
        "new_total_fraction": new_total / TOTAL_PIXELS,
        "delta_px2": new_total - old_total,
        "delta_fraction": (new_total - old_total) / TOTAL_PIXELS,
        "pct_increase": (100 * (new_total - old_total) / old_total) if old_total > 0 else 0.0,
    }



# ═════════════════════════════════════════════════════════════════════════
# PART 6 — Plot (identical to pixel_level_pointing.py, used only for
# optional spot-check PNGs, never for every pointing in a whole sim)
# ═════════════════════════════════════════════════════════════════════════

def plot_pointing(bright_streaks, normal_streaks, pixel_combined_e,
                   pixel_contributors, overlaps, meta, out_path):
    naive_xy, naive_c, combined_xy, combined_c = [], [], [], []

    for px, contributors in pixel_contributors.items():
        worst_indiv = "unsaturated"
        for si in contributors:
            st = normal_streaks[si]["individual_status"]
            if st == "faint":
                continue
            if STATUS_ORDER[st] > STATUS_ORDER[worst_indiv]:
                worst_indiv = st
        naive_xy.append(px)
        naive_c.append(PLOT_STATUSES.index(worst_indiv))

        combined_xy.append(px)
        combined_status = classify_from_electrons(pixel_combined_e[px])
        if combined_status == "faint":
            combined_status = "unsaturated"
        combined_c.append(PLOT_STATUSES.index(combined_status))

    fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharex=True, sharey=True)

    for ax in axes:
        for ccd_id, (x0, x1, y0, y1) in CCD_BBOX_PX.items():
            is_dead = any(ccd_id in b["hit_ccds"] for b in bright_streaks)
            ax.add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="black" if is_dead else "#dbe4f0",
                alpha=0.35 if is_dead else 0.6,
                edgecolor="#888888", linewidth=0.3,
            ))
        ax.add_patch(Circle(
            (0, 0), FOV_RADIUS_PX,
            facecolor="none", edgecolor="black", linewidth=1.2, zorder=5,
        ))
        for b in bright_streaks:
            ax.plot(b["wp_x"], b["wp_y"], color=STATUS_COLORS["ultrabright"],
                     linewidth=1.5, zorder=6, solid_capstyle="round")

    if len(naive_xy) > 0:
        naive_xy = np.array(naive_xy)
        axes[0].scatter(naive_xy[:, 0], naive_xy[:, 1], c=naive_c, cmap=CMAP,
                         vmin=0, vmax=2, s=1.5, rasterized=True)
    axes[0].set_title("NAIVE — each normal streak judged alone (step8 logic)")
    axes[0].set_xlabel("camera-frame x (px)")
    axes[0].set_ylabel("camera-frame y (px)")
    axes[0].set_aspect("equal")

    if len(combined_xy) > 0:
        combined_xy = np.array(combined_xy)
        axes[1].scatter(combined_xy[:, 0], combined_xy[:, 1], c=combined_c, cmap=CMAP,
                         vmin=0, vmax=2, s=1.5, rasterized=True)
    axes[1].set_title("COMBINED — overlapping normal-streak electrons summed\n(centerline-only — misses parallel/adjacent streaks)")
    axes[1].set_xlabel("camera-frame x (px)")
    axes[1].set_aspect("equal")

    draw_order = sorted(
        normal_streaks,
        key=lambda s: STATUS_ORDER.get(s["individual_status"], -1)
    )
    for s in draw_order:
        poly = s.get("mask_polygon")
        if poly is None:
            continue
        base_color = STATUS_COLORS.get(s["individual_status"], "#AAAAAA")
        base_zorder = 2 + STATUS_ORDER.get(s["individual_status"], 0)
        geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
        for g in geoms:
            xs, ys = g.exterior.xy
            axes[2].add_patch(MplPolygon(
                list(zip(xs, ys)), closed=True,
                facecolor=base_color, edgecolor="none", alpha=0.6,
                zorder=base_zorder,
            ))

    for s in draw_order:
        geom = s.get("upgraded_geom")
        if geom is None or geom.is_empty:
            continue
        worse_color = STATUS_COLORS.get(s["combined_status_mask"], "#AAAAAA")
        geoms = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for g in geoms:
            if g.geom_type != "Polygon" or g.exterior is None:
                continue
            xs, ys = g.exterior.xy
            axes[2].add_patch(MplPolygon(
                list(zip(xs, ys)), closed=True,
                facecolor=worse_color, edgecolor="none", alpha=0.8,
                zorder=10,
            ))

    for o in overlaps:
        inter = o["intersection"]
        geoms = inter.geoms if inter.geom_type == "MultiPolygon" else [inter]
        for g in geoms:
            if g.geom_type != "Polygon" or g.exterior is None:
                continue
            xs, ys = g.exterior.xy
            axes[2].add_patch(MplPolygon(
                list(zip(xs, ys)), closed=True,
                facecolor="none", edgecolor="black", linewidth=1.0,
                hatch="///", zorder=11,
            ))

    axes[2].set_title(f"MASK-POLYGON overlap (realistic — {len(overlaps)} pairs intersect)\nsolid overlap color = only the actually-upgraded sub-area; hatched = raw overlap")
    axes[2].set_xlabel("camera-frame x (px)")
    axes[2].set_aspect("equal")

    handles = [Patch(color=STATUS_COLORS[s], label=s) for s in PLOT_STATUSES[:-1]]
    handles.append(Line2D([0], [0], color=STATUS_COLORS["ultrabright"], linewidth=1.5,
                          label="ultrabright streak (ab_mag < 2.0)"))
    handles.append(Patch(color="black", alpha=0.35, label="dead CCD (bright streak)"))
    handles.append(Patch(facecolor="none", edgecolor="black", hatch="///", label="mask overlap region"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)

    all_x0 = min(min(b[0] for b in CCD_BBOX_PX.values()), -FOV_RADIUS_PX)
    all_x1 = max(max(b[1] for b in CCD_BBOX_PX.values()), FOV_RADIUS_PX)
    all_y0 = min(min(b[2] for b in CCD_BBOX_PX.values()), -FOV_RADIUS_PX)
    all_y1 = max(max(b[3] for b in CCD_BBOX_PX.values()), FOV_RADIUS_PX)
    pad = 0.03 * (all_x1 - all_x0)
    for ax in axes:
        ax.set_xlim(all_x0 - pad, all_x1 + pad)
        ax.set_ylim(all_y0 - pad, all_y1 + pad)

    n_up_centerline = sum(1 for s in normal_streaks if s.get("upgraded"))
    n_up_mask = sum(1 for s in normal_streaks if s.get("upgraded_mask"))
    fig.suptitle(
        f"pointing_id={meta['pointing_id']}  date={meta['date']}  "
        f"filter={meta['pointing_filter']}  rot_sky_pos={meta['rot_sky_pos']:.1f}°  "
        f"bright={meta['n_bright_streaks']}  normal={meta['n_normal_streaks']}  |  "
        f"upgraded — centerline: {n_up_centerline}, mask-polygon (realistic): {n_up_mask}",
        fontsize=11,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════
# PART 7 — Whole-sim runner ("step9")
# ═════════════════════════════════════════════════════════════════════════

DATE_RE = re.compile(r"streak_trajectories_(\d{4}-\d{2}-\d{2})\.csv$")

# ── Hardcoded date window: only these dates are ever processed, regardless
# of what else is found under --step2-dir. Edit these two lines directly
# for a different range. Inclusive on both ends. Plain string comparison
# works correctly here since YYYY-MM-DD zero-padded strings sort
# chronologically the same as lexicographically.
START_DATE = "2026-06-29"
END_DATE   = "2027-06-28"

SUMMARY_FIELDS = [
    "date", "pointing_id", "pointing_filter", "rot_sky_pos",
    "n_bright_streaks", "n_normal_streaks", "n_dead_ccds",
    "dead_pixel_loss_px2", "old_normal_area_px2", "new_normal_area_px2",
    "old_total_px2", "new_total_px2",
    "old_total_fraction", "new_total_fraction",
    "delta_px2", "delta_fraction", "pct_increase",
    "n_upgraded_centerline", "n_upgraded_mask", "n_overlap_pairs",
    "status", "error",
]


def discover_dates(step2_dir, dates_filter=None, max_dates=None,
                    shard_index=0, num_shards=1):
    """
    Finds every date, then optionally slices it down to one SHARD -- used
    to split the whole sim across multiple independent Slurm nodes/jobs
    (see --shard-index / --num-shards). Each shard gets a disjoint,
    round-robin subset of dates (dates[shard_index::num_shards]), so
    running all num_shards shards together covers every date exactly once
    with no overlap and no coordination needed between nodes.
    """
    paths = sorted(glob.glob(os.path.join(step2_dir, "streak_trajectories_*.csv")))
    dates = []
    for p in paths:
        m = DATE_RE.search(os.path.basename(p))
        if m:
            dates.append(m.group(1))

    # Hardcoded window applied FIRST, always -- regardless of --dates/
    # --max-dates, only dates in [START_DATE, END_DATE] are ever candidates.
    n_before_window = len(dates)
    dates = [d for d in dates if START_DATE <= d <= END_DATE]
    print(f"[INFO] Date window {START_DATE} to {END_DATE}: "
          f"keeping {len(dates)} of {n_before_window} date(s) found.")

    if dates_filter:
        wanted = set(dates_filter)
        dates = [d for d in dates if d in wanted]
    if max_dates:
        dates = dates[:max_dates]
    if num_shards > 1:
        if not (0 <= shard_index < num_shards):
            raise ValueError(f"--shard-index must be in [0, {num_shards}), got {shard_index}")
        dates = dates[shard_index::num_shards]
    return dates


def date_output_path(out_dir, date_str):
    return os.path.join(out_dir, f"pixel_loss_summary_{date_str}.csv")


def expected_pointing_count(date_str, step2_dir):
    """Cheaply counts unique pointing_ids for a date by reading ONLY the
    pointing_id column of step2 (not the full 12-column file) -- used to
    verify an existing per-date output CSV actually has one row per
    pointing, rather than being a truncated leftover from an earlier run
    that got killed mid-write. Returns None if step2 itself can't be read
    (caller falls back to a weaker check in that case)."""
    step2_path = os.path.join(step2_dir, f"streak_trajectories_{date_str}.csv")
    if not os.path.isfile(step2_path):
        return None
    try:
        ids = pd.read_csv(step2_path, usecols=["pointing_id"],
                           dtype={"pointing_id": "int32"})
        return int(ids["pointing_id"].nunique())
    except Exception:
        return None


def is_date_csv_complete(out_dir, date_str, step2_dir):
    """True only if this date's output CSV exists AND is actually
    complete -- not just present. Catches CSVs left truncated by a job
    that got killed mid-write (walltime limit, OOM, scancel, node
    failure, etc.) from BEFORE write_date_csv()'s atomic-rename fix was
    in place, or from any other source of a half-written file.

    Checks, in order:
      1. File exists and pandas can parse it at all (a truncated last
         line / corrupted CSV fails this outright).
      2. Its columns match SUMMARY_FIELDS exactly (guards against reading
         some unrelated or very old-format file).
      3. Its row count matches the number of unique pointing_ids actually
         present in that date's step2 file -- process_date() always
         writes exactly one row per pointing_id (ok, pointing_failed, or
         otherwise), so any mismatch means rows are missing.
      If step2 itself can't be read to get that expected count, falls
      back to "at least one row present" rather than blocking forever.
    """
    path = date_output_path(out_dir, date_str)
    if not os.path.isfile(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False  # unparseable / truncated -> not complete

    if list(df.columns) != SUMMARY_FIELDS:
        return False

    expected = expected_pointing_count(date_str, step2_dir)
    if expected is None:
        return len(df) > 0
    return len(df) == expected


def write_date_csv(out_dir, date_str, rows):
    """Writes one date's rows to its own CSV (overwrites if rerun).

    Writes to a temp file in the SAME directory first, then atomically
    renames it to the real path (os.replace is atomic on POSIX
    filesystems). This guarantees the final filename either doesn't exist
    at all, or exists fully-written -- never a partially-written file that
    already_done_dates() would mistake for "complete" and skip forever on
    a resumed run if the job gets killed mid-write (walltime limit, node
    failure, scancel, etc.).
    """
    os.makedirs(out_dir, exist_ok=True)
    path = date_output_path(out_dir, date_str)
    tmp_path = path + f".tmp{os.getpid()}"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})
    os.replace(tmp_path, path)  # atomic -- no partial file ever visible at `path`
    return path


def already_done_dates(out_dir, dates, step2_dir):
    """Dates whose per-date CSV already exists AND is verified complete
    (see is_date_csv_complete). A date whose file exists but is truncated
    is deliberately NOT included here, so main()'s dates-to-process list
    will include it again and process_date() will fully reprocess and
    overwrite it -- there is no partial/per-pointing resume within a
    date, a date is always redone as a whole unit if it wasn't clean."""
    done = set()
    for d in dates:
        if is_date_csv_complete(out_dir, d, step2_dir):
            done.add(d)
    return done


def analyze_one_pointing(bright_streaks, normal_streaks, meta):
    """Runs the full v2 overlap pipeline on one already-loaded pointing."""
    pixel_combined_e, pixel_contributors = accumulate_pixels(normal_streaks)
    normal_streaks = evaluate_upgrades(normal_streaks, pixel_combined_e)
    normal_streaks = build_mask_polygons(normal_streaks)
    overlaps = find_polygon_overlaps(normal_streaks)
    normal_streaks = rebuild_polygons_precise(normal_streaks)
    totals = compute_pixel_loss_totals(bright_streaks, normal_streaks)

    n_up_centerline = sum(1 for s in normal_streaks if s.get("upgraded"))
    n_up_mask = sum(1 for s in normal_streaks if s.get("upgraded_mask"))

    row = {
        "date": meta["date"],
        "pointing_id": meta["pointing_id"],
        "pointing_filter": meta["pointing_filter"],
        "rot_sky_pos": meta["rot_sky_pos"],
        "n_bright_streaks": meta["n_bright_streaks"],
        "n_normal_streaks": meta["n_normal_streaks"],
        "n_dead_ccds": totals["n_dead_ccds"],
        "dead_pixel_loss_px2": totals["dead_pixel_loss"],
        "old_normal_area_px2": totals["old_normal_area_px2"],
        "new_normal_area_px2": totals["new_normal_area_px2"],
        "old_total_px2": totals["old_total_px2"],
        "new_total_px2": totals["new_total_px2"],
        "old_total_fraction": totals["old_total_fraction"],
        "new_total_fraction": totals["new_total_fraction"],
        "delta_px2": totals["delta_px2"],
        "delta_fraction": totals["delta_fraction"],
        "pct_increase": totals["pct_increase"],
        "n_upgraded_centerline": n_up_centerline,
        "n_upgraded_mask": n_up_mask,
        "n_overlap_pairs": len(overlaps),
        "status": "ok",
        "error": "",
    }
    extras = {
        "pixel_combined_e": pixel_combined_e,
        "pixel_contributors": pixel_contributors,
        "overlaps": overlaps,
        "normal_streaks": normal_streaks,
    }
    return row, extras


def _blank_row(date_str, pointing_id, status, error):
    return {
        "date": date_str, "pointing_id": pointing_id, "pointing_filter": "",
        "rot_sky_pos": np.nan, "n_bright_streaks": 0, "n_normal_streaks": 0,
        "n_dead_ccds": 0, "dead_pixel_loss_px2": 0, "old_normal_area_px2": 0,
        "new_normal_area_px2": 0, "old_total_px2": 0, "new_total_px2": 0,
        "old_total_fraction": 0, "new_total_fraction": 0, "delta_px2": 0,
        "delta_fraction": 0, "pct_increase": 0, "n_upgraded_centerline": 0,
        "n_upgraded_mask": 0, "n_overlap_pairs": 0,
        "status": status, "error": error,
    }


def process_date(date_str, args):
    """Loads one date's data ONCE, then loops every pointing_id in it.
    Failures on an individual pointing are caught and recorded as an error
    row rather than aborting the whole date."""
    try:
        df_date = load_date_frames(date_str, args.step2_dir, args.step4b_dir)
    except Exception as e:
        return [_blank_row(date_str, -1, "date_load_failed", f"{e}")]

    sb_lookup_date = load_step6_for_date(date_str, args.step6_dir)
    pointing_ids = sorted(df_date["pointing_id"].unique().tolist())
    rot_sky_pos_by_pid = get_rot_sky_pos_batch(
        pointing_ids, opsim_db=args.opsim_db, override=args.rot_sky_pos
    )

    rows = []
    for pid in pointing_ids:
        try:
            rot_sky_pos = rot_sky_pos_by_pid.get(pid, 0.0)
            bright_streaks, normal_streaks, meta = build_pointing_from_frames(
                df_date, pid, sb_lookup_date, rot_sky_pos, date_str=date_str
            )
            row, extras = analyze_one_pointing(bright_streaks, normal_streaks, meta)

            if args.plot_dir and args.plot_all:
                out_png = os.path.join(args.plot_dir, f"pixel_level_{date_str}_pt{pid}.png")
                plot_pointing(
                    bright_streaks, normal_streaks,
                    extras["pixel_combined_e"], extras["pixel_contributors"],
                    extras["overlaps"], meta, out_png,
                )
        except Exception as e:
            row = _blank_row(date_str, pid, "pointing_failed", f"{e.__class__.__name__}: {e}")
            if args.verbose:
                print(f"[ERROR] {date_str} pointing_id={pid}: {e}", file=sys.stderr)
                traceback.print_exc()
        finally:
            # Drop the per-pointing streak lists (shapely polygons + pixel
            # dicts) as soon as we're done with them, rather than letting
            # them accumulate for the whole date -- meaningful on dates
            # with hundreds of pointings and many streaks each.
            bright_streaks = None
            normal_streaks = None
            extras = None
        rows.append(row)

    # Free this date's big DataFrame/lookup before returning, so it isn't
    # held alive any longer than necessary if this worker immediately picks
    # up another date.
    del df_date, sb_lookup_date, rot_sky_pos_by_pid
    gc.collect()

    return rows


def print_aggregate_summary(out_dir):
    """Reads every per-date CSV in out_dir back in, purely in-memory, to
    print aggregate stats -- this does NOT write a combined file; each
    date's CSV stays exactly as its own file on disk."""
    paths = sorted(glob.glob(os.path.join(out_dir, "pixel_loss_summary_*.csv")))
    if not paths:
        print(f"\n[WARN] No per-date CSVs found in {out_dir} to summarize.")
        return
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    ok = df[df["status"] == "ok"]
    failed = df[df["status"] != "ok"]

    print("\n" + "=" * 78)
    print("WHOLE-SIM AGGREGATE SUMMARY (step9)")
    print("=" * 78)
    print(f"  per-date files read : {len(paths):,}  (in {out_dir}, one file per date)")
    print(f"  pointings processed OK : {len(ok):,}")
    if len(failed):
        print(f"  pointings FAILED       : {len(failed):,}  "
              f"(see status/error columns in each date's CSV)")

    if len(ok) == 0:
        print("  No successful pointings — nothing to aggregate.")
        return

    old_sum = ok["old_total_px2"].sum()
    new_sum = ok["new_total_px2"].sum()
    dead_sum = ok["dead_pixel_loss_px2"].sum()
    n_pointings = len(ok)
    total_pixels_all = TOTAL_PIXELS * n_pointings

    print(f"  total pointings (denominator) : {n_pointings:,}")
    print(f"  OLD total pixel loss  : {old_sum:>20,.0f} px²  "
          f"({100*old_sum/total_pixels_all:.4f}% of all pointings' pixels)")
    print(f"  NEW total pixel loss  : {new_sum:>20,.0f} px²  "
          f"({100*new_sum/total_pixels_all:.4f}% of all pointings' pixels)")
    delta = new_sum - old_sum
    pct = 100 * delta / old_sum if old_sum > 0 else 0.0
    print(f"  Delta (NEW - OLD)     : {delta:>+20,.0f} px²  ({pct:+.2f}% relative increase)")
    print(f"  (of which dead-CCD/bright-streak loss, same in OLD & NEW): "
          f"{dead_sum:,.0f} px²")

    n_up = (ok["n_upgraded_mask"] > 0).sum()
    print(f"  pointings with >=1 streak upgraded by mask-polygon overlap: "
          f"{n_up:,} / {n_pointings:,} ({100*n_up/n_pointings:.1f}%)")

    print("\n  Top 10 pointings by relative delta (pct_increase):")
    top = ok.sort_values("pct_increase", ascending=False).head(10)
    for _, r in top.iterrows():
        print(f"    {r['date']}  pointing_id={int(r['pointing_id'])}  "
              f"old={r['old_total_px2']:,.0f}  new={r['new_total_px2']:,.0f}  "
              f"+{r['pct_increase']:.2f}%")


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    # ── Hardcoded defaults matching this pipeline's directory layout ──────
    ap.add_argument("--step2-dir", default="step2_output")
    ap.add_argument("--step4b-dir", default="step4b_output_rad")
    ap.add_argument("--step6-dir", default="step6_output_perband_rad")
    ap.add_argument("--opsim-db", default="baseline_v5.3.0_10yrs.db")
    ap.add_argument("--rot-sky-pos", type=float, default=None,
                     help="Override rotSkyPos for ALL pointings (skips opsim DB lookup). "
                          "Leave unset to load per-pointing from the opsim DB.")
    ap.add_argument("--out-dir", default="step9_output",
                     help="Directory to write one CSV per date into "
                          "(pixel_loss_summary_{date}.csv), rather than one shared "
                          "file across all dates. This also makes multiple shards "
                          "(see --num-shards) trivially safe to run concurrently -- "
                          "since each date already only belongs to one shard, "
                          "there's never a file two shards would both write to.")
    ap.add_argument("--dates", nargs="*", default=None,
                     help="Restrict to these YYYY-MM-DD dates only (default: all found).")
    ap.add_argument("--max-dates", type=int, default=None,
                     help="Process only the first N dates (for a quick sanity check).")
    ap.add_argument("--workers", type=int, default=12,
                     help="Parallelize across DATES (each worker loads its own date's "
                          "step2/step4b/step6 data into memory at once). Defaults to a "
                          "conservative 12 given real observed date sizes (some dates "
                          "have 500+ pointings) -- 20 workers already triggered one "
                          "OOM-kill in a 200G SLURM job. Increase gradually (e.g. "
                          "checking `seff <jobid>` or `sacct --format=MaxRSS` after a run) "
                          "rather than jumping straight back to a high number. Auto-capped "
                          "to the number of dates found if fewer than --workers.")
    ap.add_argument("--num-shards", type=int, default=1,
                     help="Split the whole sim into this many disjoint shards, so it can "
                          "be spread across multiple Slurm nodes/jobs instead of one node "
                          "with a huge --workers count. Each shard gets a round-robin "
                          "subset of dates (dates[shard_index::num_shards]) -- running "
                          "every shard 0..num_shards-1 covers all dates exactly once. "
                          "Pairs with --shard-index and $SLURM_ARRAY_TASK_ID in a job "
                          "array (see the example sbatch script).")
    ap.add_argument("--shard-index", type=int, default=0,
                     help="Which shard this run handles, in [0, num_shards). Typically "
                          "set to $SLURM_ARRAY_TASK_ID in a Slurm job array.")
    ap.add_argument("--plot-dir", default=None,
                     help="If set, save per-pointing diagnostic PNGs here.")
    ap.add_argument("--plot-all", action="store_true",
                     help="Save a plot for EVERY pointing (can be thousands of PNGs — "
                          "usually you want --plot-top-n instead).")
    ap.add_argument("--plot-top-n", type=int, default=0,
                     help="After the run, re-plot the N pointings with the largest "
                          "pct_increase, for spot-checking. Requires --plot-dir.")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main():
    args = build_arg_parser().parse_args()

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)

    dates = discover_dates(args.step2_dir, args.dates, args.max_dates,
                            shard_index=args.shard_index, num_shards=args.num_shards)
    if not dates:
        print(f"[FATAL] No streak_trajectories_*.csv files found under "
              f"{args.step2_dir!r}. Check --step2-dir.", file=sys.stderr)
        sys.exit(1)

    if args.num_shards > 1:
        print(f"[INFO] Shard {args.shard_index}/{args.num_shards}: "
              f"{len(dates)} date(s) assigned.")
    print(f"[INFO] Found {len(dates)} date(s) to process. "
          f"Writing one CSV per date into {args.out_dir}/")
    if args.workers > len(dates):
        print(f"[INFO] --workers {args.workers} exceeds {len(dates)} date(s) found — "
              f"capping to {len(dates)} (parallelism here is per-date, so extra "
              f"workers would just sit idle).")
        args.workers = max(1, len(dates))

    already_done = already_done_dates(args.out_dir, dates, args.step2_dir)
    n_files_present = sum(
        1 for d in dates if os.path.isfile(date_output_path(args.out_dir, d))
    )
    n_partial = n_files_present - len(already_done)
    if already_done:
        print(f"[INFO] Resuming — {len(already_done)} date(s) already have a "
              f"verified-complete output CSV in {args.out_dir}/, will be skipped.")
    if n_partial:
        print(f"[INFO] {n_partial} date(s) had an output CSV present but it was "
              f"incomplete/truncated (likely from a job killed mid-write) -- "
              f"these will be fully reprocessed and their file overwritten.")
    dates = [d for d in dates if d not in already_done]

    t0 = time.time()
    n_done_dates = 0
    n_total = len(dates) + len(already_done)

    def handle_date_result(date_str, rows):
        nonlocal n_done_dates
        n_done_dates += 1
        n_ok = sum(1 for r in rows if r["status"] == "ok")
        n_fail = len(rows) - n_ok
        elapsed = time.time() - t0

        # A whole-date failure (date_load_failed / date_failed, a single
        # blank row) means we never actually got this date's real data --
        # don't write a "done" marker file for it, or a resumed rerun would
        # permanently skip a date that just needs retrying.
        whole_date_failed = (
            len(rows) == 1 and rows[0]["status"] in ("date_load_failed", "date_failed")
        )
        if whole_date_failed:
            print(f"[{n_done_dates}/{len(dates)}] {date_str}: FAILED to load "
                  f"({rows[0]['error']})  -- no output written, will retry on rerun  "
                  f"({elapsed:.1f}s elapsed)")
            return

        out_path = write_date_csv(args.out_dir, date_str, rows)
        print(f"[{n_done_dates}/{len(dates)}] {date_str}: {n_ok} ok, {n_fail} failed  "
              f"-> {out_path}  ({elapsed:.1f}s elapsed)")

    if not dates:
        print("[INFO] Nothing left to do -- every date already has an output CSV.")
    elif args.workers <= 1:
        for date_str in dates:
            rows = process_date(date_str, args)
            handle_date_result(date_str, rows)
    else:
        # Self-healing pool: if a worker gets SIGKILL'd (e.g. by the OOM
        # killer), the ProcessPoolExecutor becomes permanently broken and
        # every remaining/future submission raises BrokenProcessPool
        # immediately -- without this, one OOM'd worker silently turns into
        # "every remaining date fails instantly" for the rest of the whole
        # run. Instead: on BrokenProcessPool, drop the dead pool, build a
        # fresh one, and resubmit only the dates that haven't succeeded yet.
        remaining = list(dates)
        while remaining:
            newly_done = set()
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    futures = {ex.submit(process_date, d, args): d for d in remaining}
                    for fut in as_completed(futures):
                        date_str = futures[fut]
                        try:
                            rows = fut.result()
                        except BrokenProcessPool:
                            raise
                        except Exception as e:
                            rows = [_blank_row(date_str, -1, "date_failed", f"{e}")]
                            handle_date_result(date_str, rows)
                            newly_done.add(date_str)
                            continue
                        handle_date_result(date_str, rows)
                        newly_done.add(date_str)
            except BrokenProcessPool:
                print(f"[WARN] Worker pool broken (a worker was likely killed by "
                      f"OOM). Restarting the pool and resuming with the "
                      f"{len(remaining) - len(newly_done)} date(s) not yet "
                      f"completed. Consider lowering --workers if this repeats.")
            remaining = [d for d in remaining if d not in newly_done]

    print(f"\n[INFO] Done. {n_total} date(s) total, each in its own CSV under {args.out_dir}/")
    print_aggregate_summary(args.out_dir)

    if args.plot_top_n and args.plot_dir:
        print(f"\n[INFO] Re-plotting top {args.plot_top_n} pointings by pct_increase ...")
        paths = sorted(glob.glob(os.path.join(args.out_dir, "pixel_loss_summary_*.csv")))
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()
        ok = df[df["status"] == "ok"].sort_values("pct_increase", ascending=False) if len(df) else df
        top = ok.head(args.plot_top_n)
        for _, r in top.iterrows():
            date_str, pid = str(r["date"]), int(r["pointing_id"])
            try:
                df_date = load_date_frames(date_str, args.step2_dir, args.step4b_dir)
                sb_lookup_date = load_step6_for_date(date_str, args.step6_dir)
                rot = get_rot_sky_pos_batch([pid], args.opsim_db, args.rot_sky_pos)[pid]
                bright_streaks, normal_streaks, meta = build_pointing_from_frames(
                    df_date, pid, sb_lookup_date, rot, date_str=date_str
                )
                _, extras = analyze_one_pointing(bright_streaks, normal_streaks, meta)
                out_png = os.path.join(args.plot_dir, f"pixel_level_{date_str}_pt{pid}.png")
                plot_pointing(
                    bright_streaks, extras["normal_streaks"],
                    extras["pixel_combined_e"], extras["pixel_contributors"],
                    extras["overlaps"], meta, out_png,
                )
                print(f"  [SAVED] {out_png}")
            except Exception as e:
                print(f"  [WARN] could not plot {date_str} pt{pid}: {e}")


if __name__ == "__main__":
    main()