"""
step1_streak_detection.py
─────────────────────────
Detects which Rubin/LSST pointings are affected by satellite streaks
from each SXODC shell.

METHOD
──────
For each shell × calendar day:
  1. Generate TLEs for all satellites (one per day at noon).
     Save to tles_shell{id:03d}_{YYYY-MM-DD}.pkl  (compatible with step2).
  2. Pick 1 representative satellite per orbital plane (plane[0] satellite).
  3. Propagate those rep-sats across N_STEPS time steps spanning the day (sgp4).
  4. For each pointing: if ANY rep-sat position ever falls within MATCH_DEG
     of the pointing RA/Dec → mark as hit (n_streaks=1).
     Otherwise n_streaks=0.

PARALLELISM
───────────
Shells are processed in parallel using multiprocessing.Pool.
Each shell runs in its own worker process — fully independent.
N_WORKERS = 8  (leaves 2 of your 10 cores free for the OS).

LOGGING
───────
All output goes to BOTH terminal and a log file simultaneously:
    streak_output_sgp4/step1.log

Each line is timestamped and tagged with the worker PID so you can
untangle interleaved output from parallel workers.

Watch live progress:
    tail -f streak_output_sgp4/step1.log

MUST be run as a .py file from terminal, NOT from Jupyter:
    python step1_streak_detection.py

Requirements:
    pip install sgp4
"""

import sqlite3, os, time, pickle, math, gc, logging, sys
from multiprocessing import Pool
import functools
import numpy as np
import pandas as pd
import astropy
import astropy.units as u
import astropy.time
from sgp4.api import Satrec

from rubin_sim.data import get_baseline

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str, log_filename: str = "step1.log") -> str:
    """
    Configure root logger to write to both terminal and a log file.
    Called once in main() and once at the start of each worker process.

    Returns the full path to the log file.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if called more than once
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt     = "%(asctime)s  [PID %(process)5d]  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # File handler — appends so resume runs accumulate in one log
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Terminal handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return log_path

# Convenience shorthand used throughout
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
COL_ALT     = "altitude"
COL_AZ      = "azimuth"

# ─────────────────────────────────────────────────────────────────────────────
# SXODC shell definitions
# ─────────────────────────────────────────────────────────────────────────────
SXODC_SHELLS = [
    # Layer 100–124: ~686–718 km, ~30° inclination, 30 planes × 333 sats
    {"id":  1, "layer": "LEO-30",  "alt_km":  686.0, "inc_deg": 30.5, "nplanes": 30, "sats_per_plane": 333},
    {"id":  2, "layer": "LEO-30",  "alt_km":  687.3, "inc_deg": 30.4, "nplanes": 30, "sats_per_plane": 333},
    {"id":  3, "layer": "LEO-30",  "alt_km":  688.7, "inc_deg": 30.4, "nplanes": 30, "sats_per_plane": 333},
    {"id":  4, "layer": "LEO-30",  "alt_km":  690.0, "inc_deg": 30.4, "nplanes": 30, "sats_per_plane": 333},
    {"id":  5, "layer": "LEO-30",  "alt_km":  691.3, "inc_deg": 30.3, "nplanes": 30, "sats_per_plane": 333},
    {"id":  6, "layer": "LEO-30",  "alt_km":  692.7, "inc_deg": 30.2, "nplanes": 30, "sats_per_plane": 333},
    {"id":  7, "layer": "LEO-30",  "alt_km":  694.0, "inc_deg": 30.2, "nplanes": 30, "sats_per_plane": 333},
    {"id":  8, "layer": "LEO-30",  "alt_km":  695.3, "inc_deg": 30.2, "nplanes": 30, "sats_per_plane": 333},
    {"id":  9, "layer": "LEO-30",  "alt_km":  696.7, "inc_deg": 30.1, "nplanes": 30, "sats_per_plane": 333},
    {"id": 10, "layer": "LEO-30",  "alt_km":  698.0, "inc_deg": 30.1, "nplanes": 30, "sats_per_plane": 333},
    {"id": 11, "layer": "LEO-30",  "alt_km":  699.3, "inc_deg": 30.0, "nplanes": 30, "sats_per_plane": 333},
    {"id": 12, "layer": "LEO-30",  "alt_km":  700.7, "inc_deg": 30.0, "nplanes": 30, "sats_per_plane": 333},
    {"id": 13, "layer": "LEO-30",  "alt_km":  702.0, "inc_deg": 29.9, "nplanes": 30, "sats_per_plane": 333},
    {"id": 14, "layer": "LEO-30",  "alt_km":  703.3, "inc_deg": 29.9, "nplanes": 30, "sats_per_plane": 333},
    {"id": 15, "layer": "LEO-30",  "alt_km":  704.7, "inc_deg": 29.8, "nplanes": 30, "sats_per_plane": 333},
    {"id": 16, "layer": "LEO-30",  "alt_km":  706.0, "inc_deg": 29.8, "nplanes": 30, "sats_per_plane": 333},
    {"id": 17, "layer": "LEO-30",  "alt_km":  707.3, "inc_deg": 29.7, "nplanes": 30, "sats_per_plane": 333},
    {"id": 18, "layer": "LEO-30",  "alt_km":  708.7, "inc_deg": 29.7, "nplanes": 30, "sats_per_plane": 333},
    {"id": 19, "layer": "LEO-30",  "alt_km":  710.0, "inc_deg": 29.6, "nplanes": 30, "sats_per_plane": 333},
    {"id": 20, "layer": "LEO-30",  "alt_km":  711.3, "inc_deg": 29.6, "nplanes": 30, "sats_per_plane": 333},
    {"id": 21, "layer": "LEO-30",  "alt_km":  712.7, "inc_deg": 29.5, "nplanes": 30, "sats_per_plane": 333},
    {"id": 22, "layer": "LEO-30",  "alt_km":  714.0, "inc_deg": 29.5, "nplanes": 30, "sats_per_plane": 333},
    {"id": 23, "layer": "LEO-30",  "alt_km":  715.3, "inc_deg": 29.4, "nplanes": 30, "sats_per_plane": 333},
    {"id": 24, "layer": "LEO-30",  "alt_km":  716.7, "inc_deg": 29.4, "nplanes": 30, "sats_per_plane": 333},
    {"id": 25, "layer": "LEO-30",  "alt_km":  718.0, "inc_deg": 29.3, "nplanes": 30, "sats_per_plane": 333},
    # Layer 200–224: ~946–978 km, ~30° inclination, 30 planes × 333 sats
    {"id": 26, "layer": "MEO-30",  "alt_km":  946.0, "inc_deg": 30.5, "nplanes": 30, "sats_per_plane": 333},
    {"id": 27, "layer": "MEO-30",  "alt_km":  947.3, "inc_deg": 30.4, "nplanes": 30, "sats_per_plane": 333},
    {"id": 28, "layer": "MEO-30",  "alt_km":  948.7, "inc_deg": 30.4, "nplanes": 30, "sats_per_plane": 333},
    {"id": 29, "layer": "MEO-30",  "alt_km":  950.0, "inc_deg": 30.4, "nplanes": 30, "sats_per_plane": 333},
    {"id": 30, "layer": "MEO-30",  "alt_km":  951.3, "inc_deg": 30.3, "nplanes": 30, "sats_per_plane": 333},
    {"id": 31, "layer": "MEO-30",  "alt_km":  952.7, "inc_deg": 30.3, "nplanes": 30, "sats_per_plane": 333},
    {"id": 32, "layer": "MEO-30",  "alt_km":  954.0, "inc_deg": 30.2, "nplanes": 30, "sats_per_plane": 333},
    {"id": 33, "layer": "MEO-30",  "alt_km":  955.3, "inc_deg": 30.2, "nplanes": 30, "sats_per_plane": 333},
    {"id": 34, "layer": "MEO-30",  "alt_km":  956.7, "inc_deg": 30.1, "nplanes": 30, "sats_per_plane": 333},
    {"id": 35, "layer": "MEO-30",  "alt_km":  958.0, "inc_deg": 30.1, "nplanes": 30, "sats_per_plane": 333},
    {"id": 36, "layer": "MEO-30",  "alt_km":  959.3, "inc_deg": 30.0, "nplanes": 30, "sats_per_plane": 333},
    {"id": 37, "layer": "MEO-30",  "alt_km":  960.7, "inc_deg": 30.0, "nplanes": 30, "sats_per_plane": 333},
    {"id": 38, "layer": "MEO-30",  "alt_km":  962.0, "inc_deg": 29.9, "nplanes": 30, "sats_per_plane": 333},
    {"id": 39, "layer": "MEO-30",  "alt_km":  963.3, "inc_deg": 29.9, "nplanes": 30, "sats_per_plane": 333},
    {"id": 40, "layer": "MEO-30",  "alt_km":  964.7, "inc_deg": 29.8, "nplanes": 30, "sats_per_plane": 333},
    {"id": 41, "layer": "MEO-30",  "alt_km":  966.0, "inc_deg": 29.8, "nplanes": 30, "sats_per_plane": 333},
    {"id": 42, "layer": "MEO-30",  "alt_km":  967.3, "inc_deg": 29.7, "nplanes": 30, "sats_per_plane": 333},
    {"id": 43, "layer": "MEO-30",  "alt_km":  968.7, "inc_deg": 29.7, "nplanes": 30, "sats_per_plane": 333},
    {"id": 44, "layer": "MEO-30",  "alt_km":  970.0, "inc_deg": 29.6, "nplanes": 30, "sats_per_plane": 333},
    {"id": 45, "layer": "MEO-30",  "alt_km":  971.3, "inc_deg": 29.6, "nplanes": 30, "sats_per_plane": 333},
    {"id": 46, "layer": "MEO-30",  "alt_km":  972.7, "inc_deg": 29.6, "nplanes": 30, "sats_per_plane": 333},
    {"id": 47, "layer": "MEO-30",  "alt_km":  974.0, "inc_deg": 29.5, "nplanes": 30, "sats_per_plane": 333},
    {"id": 48, "layer": "MEO-30",  "alt_km":  975.3, "inc_deg": 29.4, "nplanes": 30, "sats_per_plane": 333},
    {"id": 49, "layer": "MEO-30",  "alt_km":  976.7, "inc_deg": 29.4, "nplanes": 30, "sats_per_plane": 333},
    {"id": 50, "layer": "MEO-30",  "alt_km":  978.0, "inc_deg": 29.4, "nplanes": 30, "sats_per_plane": 333},
    # Layer 300–321: ~707–744 km, ~97–98° inclination (SSO), 1 plane × 11,131 sats
    {"id": 51, "layer": "SSO-97",  "alt_km":  707.0, "inc_deg": 97.1, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 52, "layer": "SSO-97",  "alt_km":  708.8, "inc_deg": 97.2, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 53, "layer": "SSO-97",  "alt_km":  710.5, "inc_deg": 97.2, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 54, "layer": "SSO-97",  "alt_km":  712.3, "inc_deg": 97.3, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 55, "layer": "SSO-97",  "alt_km":  714.0, "inc_deg": 97.3, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 56, "layer": "SSO-97",  "alt_km":  715.8, "inc_deg": 97.4, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 57, "layer": "SSO-97",  "alt_km":  717.6, "inc_deg": 97.4, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 58, "layer": "SSO-97",  "alt_km":  719.3, "inc_deg": 97.5, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 59, "layer": "SSO-97",  "alt_km":  721.1, "inc_deg": 97.5, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 60, "layer": "SSO-97",  "alt_km":  722.9, "inc_deg": 97.6, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 61, "layer": "SSO-97",  "alt_km":  724.6, "inc_deg": 97.7, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 62, "layer": "SSO-97",  "alt_km":  726.4, "inc_deg": 97.7, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 63, "layer": "SSO-97",  "alt_km":  728.1, "inc_deg": 97.8, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 64, "layer": "SSO-97",  "alt_km":  729.9, "inc_deg": 97.8, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 65, "layer": "SSO-97",  "alt_km":  731.7, "inc_deg": 97.9, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 66, "layer": "SSO-97",  "alt_km":  733.4, "inc_deg": 97.9, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 67, "layer": "SSO-97",  "alt_km":  735.2, "inc_deg": 98.0, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 68, "layer": "SSO-97",  "alt_km":  737.0, "inc_deg": 98.0, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 69, "layer": "SSO-97",  "alt_km":  738.7, "inc_deg": 98.1, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 70, "layer": "SSO-97",  "alt_km":  740.5, "inc_deg": 98.1, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 71, "layer": "SSO-97",  "alt_km":  742.2, "inc_deg": 98.2, "nplanes":  1, "sats_per_plane": 11131},
    {"id": 72, "layer": "SSO-97",  "alt_km":  744.0, "inc_deg": 98.2, "nplanes":  1, "sats_per_plane": 11131},
    # Layer 400–421: ~967–1002 km, ~99.4–99.5° inclination (SSO), 1 plane × 11,539 sats
    {"id": 73, "layer": "SSO-99",  "alt_km":  967.0, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 74, "layer": "SSO-99",  "alt_km":  968.7, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 75, "layer": "SSO-99",  "alt_km":  970.3, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 76, "layer": "SSO-99",  "alt_km":  972.0, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 77, "layer": "SSO-99",  "alt_km":  973.7, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 78, "layer": "SSO-99",  "alt_km":  975.3, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 79, "layer": "SSO-99",  "alt_km":  977.0, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 80, "layer": "SSO-99",  "alt_km":  978.7, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 81, "layer": "SSO-99",  "alt_km":  980.3, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 82, "layer": "SSO-99",  "alt_km":  982.0, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 83, "layer": "SSO-99",  "alt_km":  983.7, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 84, "layer": "SSO-99",  "alt_km":  985.3, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 85, "layer": "SSO-99",  "alt_km":  987.0, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 86, "layer": "SSO-99",  "alt_km":  988.7, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 87, "layer": "SSO-99",  "alt_km":  990.3, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 88, "layer": "SSO-99",  "alt_km":  992.0, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 89, "layer": "SSO-99",  "alt_km":  993.7, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 90, "layer": "SSO-99",  "alt_km":  995.3, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 91, "layer": "SSO-99",  "alt_km":  997.0, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 92, "layer": "SSO-99",  "alt_km":  998.7, "inc_deg": 99.4, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 93, "layer": "SSO-99",  "alt_km": 1000.3, "inc_deg": 99.5, "nplanes":  1, "sats_per_plane": 11539},
    {"id": 94, "layer": "SSO-99",  "alt_km": 1002.0, "inc_deg": 99.5, "nplanes":  1, "sats_per_plane": 11539},
]

# ─────────────────────────────────────────────────────────────────────────────
# TLE generation
# ─────────────────────────────────────────────────────────────────────────────

def mjd_noon(mjd: float) -> float:
    return math.floor(mjd) + 0.5

def mjd_to_tle_epoch(mjd):
    t          = astropy.time.Time(mjd, format='mjd', scale='utc')
    epoch_year = int(t.strftime('%y'))
    doy        = int(t.strftime('%j'))
    midnight   = astropy.time.Time(t.strftime('%Y-%m-%d'), format='iso', scale='utc')
    frac_day   = (t - midnight).to(u.day).value
    return epoch_year, doy + frac_day

def _tle_checksum(line):
    return sum(int(c) if c.isdigit() else (1 if c == '-' else 0)
               for c in line[:68]) % 10

def make_tles_for_shell(shell: dict, mjd_epoch: float, seed: int = 42) -> list:
    epoch_year, epoch_day = mjd_to_tle_epoch(mjd_epoch)
    MU, RE   = 398600.4418, 6378.137
    alt      = shell["alt_km"]
    inc      = shell["inc_deg"]
    np_      = shell["nplanes"]
    spp      = shell["sats_per_plane"]
    shell_id = shell.get("id", 1)

    n_rev  = (86400.0 / (2 * np.pi)) * np.sqrt(MU / (RE + alt) ** 3)
    raans  = np.linspace(0, 360, np_, endpoint=False)
    rng    = np.random.default_rng(seed)

    lines = []
    for plane_i in range(np_):
        m0s   = (np.linspace(0, 360, spp, endpoint=False)
                 + rng.uniform(0, 360 / spp)) % 360
        argps = rng.uniform(0, 360, spp)
        for j in range(spp):
            sid  = plane_i * spp + j + 1
            name = f"SH{shell_id:02d}-{plane_i:03d}-{j:03d}"
            l1_body = (f"1 {sid:05d}U 24001{sid % 1000:03d}A "
                       f"{epoch_year:02d}{epoch_day:012.8f} "
                       f" .00000100  00000-0  15000-4 0  999")
            l1_body = l1_body[:68].ljust(68)
            l1 = l1_body + str(_tle_checksum(l1_body))
            l2_body = (f"2 {sid:05d} {inc:8.4f} {raans[plane_i]:8.4f} "
                       f"0001000 {argps[j]:8.4f} "
                       f"{m0s[j]:8.4f} {n_rev:11.8f}    10")
            l2_body = l2_body[:68].ljust(68)
            l2 = l2_body + str(_tle_checksum(l2_body))
            lines.append(f"{name}\n{l1}\n{l2}")
    return lines

# ─────────────────────────────────────────────────────────────────────────────
# Load pointings
# ─────────────────────────────────────────────────────────────────────────────

def load_opsim_pointings(db_path=None, night_min=None, night_max=None,
                         mjd_min=None, mjd_max=None, filters=None,
                         max_rows=None, sql_extra=None):
    if db_path is None:
        db_path = get_baseline()
        log(f"[OPSIM] Database : {db_path}")
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Not found: {db_path}")

    with sqlite3.connect(db_path) as con:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", con
        )["name"].tolist()

    TABLE = next((t for t in ["observations","SummaryAllProps","Summary"]
                  if t in tables), None)
    if TABLE is None:
        raise RuntimeError(f"No observations table. Available: {tables}")
    log(f"[OPSIM] Table    : '{TABLE}'")

    cols = [COL_ID, COL_RA, COL_DEC, COL_MJD, COL_EXPTIME,
            COL_FILTER, COL_NIGHT, COL_ALT, COL_AZ]
    wheres = []
    if night_min is not None: wheres.append(f"{COL_NIGHT} >= {night_min}")
    if night_max is not None: wheres.append(f"{COL_NIGHT} <= {night_max}")
    if mjd_min   is not None: wheres.append(f"{COL_MJD} >= {mjd_min}")
    if mjd_max   is not None: wheres.append(f"{COL_MJD} <= {mjd_max}")
    if filters:
        wheres.append(f"{COL_FILTER} IN ({', '.join(repr(f) for f in filters)})")
    if sql_extra:
        wheres.append(f"({sql_extra})")

    where_sql = ("WHERE " + " AND ".join(wheres)) if wheres else ""
    limit_sql = f"LIMIT {max_rows}" if max_rows else ""
    query = (f"SELECT {', '.join(cols)} FROM {TABLE} "
             f"{where_sql} ORDER BY {COL_MJD} {limit_sql}")
    log(f"[OPSIM] Query    : {query[:180]}{'...' if len(query)>180 else ''}")

    with sqlite3.connect(db_path) as con:
        df = pd.read_sql(query, con)

    df["mjd_noon"] = df[COL_MJD].apply(mjd_noon)
    log(f"[OPSIM] Loaded   : {len(df):,} pointings  "
        f"MJD {df[COL_MJD].min():.4f} – {df[COL_MJD].max():.4f}  "
        f"({df['mjd_noon'].nunique()} unique calendar days)")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# sgp4 propagation helpers
# ─────────────────────────────────────────────────────────────────────────────

def mjd_to_jd_split(mjd: np.ndarray):
    jd_full = mjd + 2400000.5
    jd1     = np.floor(jd_full).astype(np.float64)
    jd2     = (jd_full - jd1).astype(np.float64)
    return jd1, jd2

def _gmst_rad(jd_ut1: np.ndarray) -> np.ndarray:
    T        = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = (67310.54841
                + (876600.0 * 3600.0 + 8640184.812866) * T
                + 0.093104 * T**2
                - 6.2e-6   * T**3)
    return np.radians(gmst_sec % 86400.0 / 240.0)

def teme_to_radec_vectorised(r_teme: np.ndarray, jd_full: np.ndarray):
    theta  = _gmst_rad(jd_full)
    ct, st = np.cos(theta), np.sin(theta)
    x_t = r_teme[:, 0];  y_t = r_teme[:, 1];  z_t = r_teme[:, 2]
    x_g =  ct * x_t + st * y_t
    y_g = -st * x_t + ct * y_t
    z_g =  z_t
    rr  = np.sqrt(x_g**2 + y_g**2 + z_g**2)
    ra  = (np.degrees(np.arctan2(y_g, x_g)) % 360.0).astype(np.float32)
    dec = np.degrees(np.arcsin(np.clip(z_g / rr, -1.0, 1.0))).astype(np.float32)
    return ra, dec

# ─────────────────────────────────────────────────────────────────────────────
# Angular separation  (haversine, vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def ang_sep_deg_vec(ra1, dec1, ra2, dec2):
    r1 = np.radians(ra1);  d1 = np.radians(dec1)
    r2 = np.radians(ra2);  d2 = np.radians(dec2)
    dra  = r1 - r2
    ddec = d1 - d2
    a    = np.sin(ddec/2)**2 + np.cos(d1)*np.cos(d2)*np.sin(dra/2)**2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))

# ─────────────────────────────────────────────────────────────────────────────
# CORE: check all pointings for one shell × one day
# ─────────────────────────────────────────────────────────────────────────────

def check_shell_day(
    tle_lines:      list,
    shell:          dict,
    day_pts:        pd.DataFrame,
    noon_mjd:       float,
    n_steps:        int   = 1500,
    match_deg:      float = 5.0,
    fov_radius_deg: float = 1.75,
) -> pd.DataFrame:
    nplanes = shell["nplanes"]
    spp     = shell["sats_per_plane"]

    mjd_grid = np.linspace(noon_mjd - 0.5, noon_mjd + 0.5, n_steps)
    jd1, jd2 = mjd_to_jd_split(mjd_grid)
    jd_full   = jd1 + jd2

    all_ra_list  = []
    all_dec_list = []
    n_failed     = 0

    for plane_i in range(nplanes):
        rep_idx = plane_i * spp
        tle     = tle_lines[rep_idx]
        parts   = tle.strip().split("\n")
        if len(parts) != 3:
            n_failed += 1
            continue
        _, l1, l2 = parts
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

        ra, dec = teme_to_radec_vectorised(r, jd_full)
        all_ra_list.append(ra[ok])
        all_dec_list.append(dec[ok])

    if not all_ra_list:
        out = day_pts.copy()
        out["n_streaks"]         = 0
        out["streak_length_deg"] = 0.0
        return out

    cloud_ra  = np.concatenate(all_ra_list)
    cloud_dec = np.concatenate(all_dec_list)

    pt_ra  = day_pts[COL_RA].values.astype(np.float32)
    pt_dec = day_pts[COL_DEC].values.astype(np.float32)
    N      = len(day_pts)

    CHUNK   = 50_000
    min_sep = np.full(N, 999.0, dtype=np.float32)

    for start in range(0, len(cloud_ra), CHUNK):
        c_ra  = cloud_ra [start:start+CHUNK]
        c_dec = cloud_dec[start:start+CHUNK]
        seps  = ang_sep_deg_vec(
            pt_ra [:, None], pt_dec[:, None],
            c_ra  [None, :], c_dec [None, :],
        )
        min_sep = np.minimum(min_sep, seps.min(axis=1))

    n_streaks         = np.zeros(N, dtype=int)
    streak_length_deg = np.zeros(N, dtype=float)
    hit_mask          = (min_sep <= match_deg)
    n_streaks[hit_mask]         = 1
    streak_length_deg[hit_mask] = 2.0 * fov_radius_deg

    out = day_pts.copy()
    out["n_streaks"]         = n_streaks
    out["streak_length_deg"] = streak_length_deg
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Worker function — processes ONE shell across ALL days
# Must be top-level (not nested) for Mac multiprocessing spawn to pickle it
# ─────────────────────────────────────────────────────────────────────────────

def process_shell(shell, pointings, unique_noons, output_dir,
                  match_deg, n_steps, fov_radius_deg, log_path):
    """
    Process one shell across all calendar days.
    Runs in a worker process.
    Returns the shell DataFrame (small — safe to pickle back to main).
    """
    # Re-initialise logging in this worker process
    # (each spawned process starts fresh with no handlers)
    setup_logging(output_dir, os.path.basename(log_path))

    shell_id  = shell["id"]
    shell_tag = f"shell_{shell_id:03d}"
    out_csv   = os.path.join(output_dir, f"streak_results_{shell_tag}.csv")

    # Resume-friendly
    if os.path.isfile(out_csv):
        log(f"[SKIP] Shell {shell_id:03d} — {out_csv} exists.")
        return pd.read_csv(out_csv)

    log("─" * 60)
    log(f"SHELL {shell_id:03d}  |  alt={shell['alt_km']} km  "
        f"inc={shell['inc_deg']}°  "
        f"{shell['nplanes']}×{shell['sats_per_plane']}  "
        f"[{shell['layer']}]")

    day_results   = []
    t_shell_start = time.perf_counter()

    for noon_mjd in unique_noons:
        day_mask = pointings["mjd_noon"] == noon_mjd
        day_pts  = pointings[day_mask].copy()
        if day_pts.empty:
            continue

        t_label = astropy.time.Time(
            noon_mjd, format='mjd', scale='utc'
        ).iso[:10]
        tle_pkl = os.path.join(
            output_dir, f"tles_shell{shell_id:03d}_{t_label}.pkl"
        )

        # ── Generate or load TLEs ──────────────────────────────────────────
        if os.path.isfile(tle_pkl):
            with open(tle_pkl, "rb") as f:
                cached = pickle.load(f)
            tle_lines = cached["tle_lines"] if isinstance(cached, dict) else cached
            log(f"  S{shell_id:03d} {t_label} | TLEs loaded from cache "
                f"({len(tle_lines):,} TLEs)")
        else:
            day_seed  = shell_id * 10_000 + int(noon_mjd)
            t0        = time.perf_counter()
            tle_lines = make_tles_for_shell(
                shell, mjd_epoch=noon_mjd, seed=day_seed
            )
            with open(tle_pkl, "wb") as f:
                pickle.dump({"tle_lines": tle_lines}, f)
            log(f"  S{shell_id:03d} {t_label} | TLEs generated "
                f"({len(tle_lines):,} TLEs)  "
                f"gen={time.perf_counter()-t0:.2f}s")

        # ── Check pointings ────────────────────────────────────────────────
        t0      = time.perf_counter()
        day_out = check_shell_day(
            tle_lines      = tle_lines,
            shell          = shell,
            day_pts        = day_pts,
            noon_mjd       = noon_mjd,
            n_steps        = n_steps,
            match_deg      = match_deg,
            fov_radius_deg = fov_radius_deg,
        )
        elapsed = time.perf_counter() - t0
        n_hit   = int((day_out["n_streaks"] > 0).sum())
        pct_hit = 100 * n_hit / max(len(day_out), 1)

        log(f"  S{shell_id:03d} {t_label} | "
            f"hits={n_hit:,}/{len(day_out):,} ({pct_hit:.1f}%)  "
            f"check={_hms(elapsed)}")

        day_results.append(day_out)
        del tle_lines
        gc.collect()

    if not day_results:
        log(f"[WARN] Shell {shell_id:03d} — no results!")
        return pd.DataFrame()

    # ── Combine days ───────────────────────────────────────────────────────
    shell_df = (pd.concat(day_results, ignore_index=True)
                  .sort_values(COL_MJD)
                  .reset_index(drop=True))

    shell_df["shell_id"]      = shell_id
    shell_df["shell_layer"]   = shell["layer"]
    shell_df["shell_alt_km"]  = shell["alt_km"]
    shell_df["shell_inc_deg"] = shell["inc_deg"]

    shell_df.to_csv(out_csv, index=False)

    total_time = time.perf_counter() - t_shell_start
    n_hits_total = int((shell_df["n_streaks"] > 0).sum())
    log(f"[SAVED] Shell {shell_id:03d} → {out_csv}  "
        f"rows={len(shell_df):,}  hits={n_hits_total:,}  "
        f"total={_hms(total_time)}")

    return shell_df

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main(
    db_path        = None,
    night_min      = 1,
    night_max      = 3,
    filters        = None,
    max_pointings  = None,
    shells         = None,
    fov_radius_deg = 1.75,
    match_deg      = 5.0,
    n_steps        = 1500,
    n_workers      = 8,
    output_dir     = "streak_output_sgp4",
    results_csv    = "streak_results_all.csv",
    log_filename   = "step1.log",
):
    # ── Set up logging (main process) ───────────────────────────────────────
    log_path = setup_logging(output_dir, log_filename)

    wall_start = time.perf_counter()
    _TIMINGS.clear()

    log("=" * 70)
    log("  STEP 1: Streak Detection  (sgp4, 1 rep-sat per plane)")
    log("=" * 70)
    log(f"  match_deg      : {match_deg}°")
    log(f"  n_steps        : {n_steps}")
    log(f"  n_workers      : {n_workers}")
    log(f"  fov_radius_deg : {fov_radius_deg}°")
    log(f"  output_dir     : {output_dir}")
    log(f"  log file       : {log_path}")

    if shells is None:
        shells = SXODC_SHELLS

    os.makedirs(output_dir, exist_ok=True)

    # ── Load pointings ──────────────────────────────────────────────────────
    log("")
    log("[STEP 1] Loading opsim pointings …")
    with Timer("load pointings"):
        pointings = load_opsim_pointings(
            db_path   = db_path,
            night_min = night_min,
            night_max = night_max,
            filters   = filters,
            max_rows  = max_pointings,
        )

    unique_noons = sorted(pointings["mjd_noon"].unique())
    log(f"[INFO] {len(unique_noons)} calendar day(s): "
        f"{unique_noons[0]:.1f} … {unique_noons[-1]:.1f}")

    # ── Dispatch shells to parallel workers ────────────────────────────────
    log("")
    log(f"[INFO] Dispatching {len(shells)} shells to "
        f"{n_workers} parallel workers …")
    log(f"[INFO] Worker output interleaved below — use PID column to sort")
    log("")

    worker_fn = functools.partial(
        process_shell,
        pointings    = pointings,
        unique_noons = unique_noons,
        output_dir   = output_dir,
        match_deg    = match_deg,
        n_steps      = n_steps,
        fov_radius_deg = fov_radius_deg,
        log_path     = log_path,
    )

    with Timer("parallel shell processing"):
        with Pool(processes=n_workers) as pool:
            results = pool.map(worker_fn, shells)

    all_shell_results = [r for r in results if r is not None and not r.empty]

    # ── Merge all shell results ─────────────────────────────────────────────
    log("")
    log("=" * 70)
    log("  Merging all shell results …")

    if all_shell_results:
        merged = pd.concat(all_shell_results, ignore_index=True)

        log("")
        log("--- Per-shell summary ---")
        summary = (
            merged.groupby(["shell_id","shell_layer","shell_alt_km","shell_inc_deg"])
            .agg(
                n_pointings   = (COL_ID,      "count"),
                n_hit         = ("n_streaks",  lambda x: (x>0).sum()),
                total_streaks = ("n_streaks",  "sum"),
            ).reset_index()
        )
        # Log summary line by line so it appears in the log file
        for line in summary.to_string(index=False).split("\n"):
            log(line)

        with Timer("save merged CSV"):
            merged_path = os.path.join(output_dir, results_csv)
            merged.to_csv(merged_path, index=False)
        log(f"[SAVED] {merged_path}  ({len(merged):,} rows)")
    else:
        log("[WARN] No results to merge!")
        merged = None

    _TIMINGS["TOTAL WALL TIME"] = time.perf_counter() - wall_start
    log("")
    log_timing_summary()
    log("")
    log("[DONE] — now run step2_propagate_and_match.py")
    log(f"[LOG]  Full log saved to: {log_path}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main(
        db_path        = None,
        night_min      = 365,
        night_max      = 729,
        filters        = None,
        max_pointings  = None,
        shells         = SXODC_SHELLS,
        fov_radius_deg = 1.75,
        match_deg      = 5.0,
        n_steps        = 1500,
        n_workers      = 40,
        output_dir     = "streak_output_sgp4",
        results_csv    = "streak_results_all.csv",
        log_filename   = "step1.log",
    )