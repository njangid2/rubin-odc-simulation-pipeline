"""
step10_night_sky_brightness.py
─────────────────────────────────────────────────────────────────────────────
Computes the WHOLE-SKY, WHOLE-CONSTELLATION increase in night sky brightness
from the SXODC constellation, for one night at Rubin Observatory.

This is a DIFFERENT quantity from the step9 pixel-loss pipeline: pixel loss
is about streaks crossing LSST's narrow ~9.6 deg^2 exposure FOV during
actual exposures. This script is about the DIFFUSE background brightness
added to the WHOLE SKY by every satellite that's above the horizon and
sunlit at any given moment, regardless of whether a telescope is looking
at it.

METHOD (grounded in your actual step1/step2/step3/step4 code -- nothing
guessed)
────────────────────────────────────────────────────────────────────────
For a chosen night (astronomical night: sun_el_deg <= -18 deg) and a chosen
time step:
    1. For EVERY one of the 94 SXODC shells, load that shell's FULL
       constellation TLE set for this date from the cached
       tles_shell{id:03d}_{date}.pkl files written by step1 (NOT the
       "1 rep-sat per plane" subset step1's own pointing pre-filter uses --
       every real satellite in the shell).
    2. Propagate every satellite in every shell to this timestep with sgp4
       (Satrec.twoline2rv + sgp4_array), exactly as step1/step2 do.
    3. TEME -> GCRS via the same GMST rotation as step2's teme_to_gcrs().
    4. Determine sunlit (eclipse) status via the EXACT cylindrical
       Earth-shadow model from step2's compute_sunlit() (same low-precision
       solar-position formula, same RE=6378.137 km, same proj/r2 test).
    5. GCRS -> RA/Dec via step2's gcrs_to_radec(), then RA/Dec -> topocentric
       Alt/Az at Rubin Observatory via the same astropy transform step3a
       uses (same site: lat=-30.244639, lon=-70.749417, elev=2663.0 m).
    6. Keep only satellites that are sunlit AND above --elevation-cutoff-deg
       (default 0 deg = above horizon; raise this if you want to match a
       specific "usable sky" convention).
    7. Get the Sun's Alt/Az at Rubin for this timestep the same way step3b
       does (astropy get_sun() -> AltAz).
    8. For EACH shell separately (calculate_brightness's sat_height is a
       single scalar broadcast to every satellite in one call -- shells
       have different orbital altitudes, so they can't be mixed in one
       call), call data_center_20deg.calculate_brightness() ONCE, vectorised
       over every currently-visible+sunlit satellite in that shell, with
       the SAME power_kw=400.0, continuous=False, offset_deg=20 used
       throughout your existing pipeline (step4_brightness_20deg.py).
    9. Sum flux from every satellite across every shell at this timestep:
           F_total = sum(10^(-0.4 * ab_mag_i))
       and express the added surface brightness as:
           mu_added = -2.5*log10(F_total / Omega)
       where Omega is the solid angle of the sky region being reported
       (whole hemisphere above the elevation cutoff, in arcsec^2).
   10. Combine with a natural-sky baseline to get the fractional/mag
       increase. Two modes (--sky-baseline):
           "fixed" (default, original behavior): one constant
               (default 21.8 mag/arcsec^2, roughly the 532nm/V-band
               dark-sky value) for the whole night and whole sky:
                   F_natural = 10^(-0.4*mu_natural)
                   mu_combined = -2.5*log10(F_natural + F_total/Omega)
                   delta_mag = mu_natural - mu_combined  (+ = brighter)
                   pct_increase = 100 * (F_total/Omega) / F_natural
           "real": the actual per-timestep, per-band sky brightness
               forecast from rubin_scheduler.skybrightness_pre.SkyModelPre
               (the same engine the Rubin scheduler itself queries; see
               RTN-012, https://rtn-012.lsst.io/), binned onto a healpix
               grid alongside the satellite flux so the comparison is
               area-weighted and varies with moon phase/position and
               twilight instead of a single flat number. See
               RealSkyBaseline's docstring for requirements and caveats.

Repeats across every timestep in the night, reporting the time series plus
peak/mean summary values.

WHAT'S VERIFIED VS. WHAT NEEDS YOUR REAL ENVIRONMENT
───────────────────────────────────────────────────────
Everything EXCEPT the call to data_center_20deg.calculate_brightness() has
been tested end-to-end against synthetic TLEs in a sandbox (orbital
propagation, eclipse determination, Alt/Az conversion, Sun position, flux
summing, and the surrounding aggregation logic). The actual brightness call
depends on lumos/starlink/analysis packages that only exist in your real
environment -- see BrightnessBackend below for how to wire it in, and the
included MockBrightnessBackend for what was used to validate the rest of
the pipeline without those packages.

INPUTS
──────
    streak_output_sgp4/tles_shell{id:03d}_{YYYY-MM-DD}.pkl   (from step1)
    data_center_20deg.calculate_brightness   (your real brightness model)

USAGE
─────
    python3 step10_night_sky_brightness.py --date 2026-06-29

    # narrower elevation cutoff, coarser time step for a quick check:
    python3 step10_night_sky_brightness.py --date 2026-06-29 \\
        --elevation-cutoff-deg 20 --time-step-sec 300

    # real per-timestep/per-position sky brightness instead of a fixed
    # 21.8 mag/arcsec^2 constant (needs rubin_scheduler + its data files):
    python3 step10_night_sky_brightness.py --date 2026-06-29 \\
        --sky-baseline real --band r

    # real, actually-simulated per-visit sky brightness pulled straight out
    # of an opsim run (no extra package/data product needed -- just the
    # .db file itself):
    python3 step10_night_sky_brightness.py --date 2026-06-29 \\
        --sky-baseline opsim --opsim-db baseline_v5.3.0_10yrs.db --band r
"""

import argparse
import glob
import math
import os
import pickle
import sqlite3
import sys

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────
# Rubin Observatory (identical constants to step3a.py / step3b.py)
# ─────────────────────────────────────────────────────────────────────────

RUBIN_LAT_DEG = -30.244639
RUBIN_LON_DEG = -70.749417
RUBIN_ELEV_M = 2663.0

# ─────────────────────────────────────────────────────────────────────────
# SXODC shell definitions -- altitude only needed here (for sat_height and
# grouping calculate_brightness calls by shell). Full shell dicts (nplanes,
# sats_per_plane, inc_deg) aren't needed since we read TLEs directly from
# the cached .pkl files rather than regenerating them.
# ─────────────────────────────────────────────────────────────────────────

SHELL_ALT_KM = {
    1: 686.0, 2: 687.3, 3: 688.7, 4: 690.0, 5: 691.3, 6: 692.7, 7: 694.0,
    8: 695.3, 9: 696.7, 10: 698.0, 11: 699.3, 12: 700.7, 13: 702.0,
    14: 703.3, 15: 704.7, 16: 706.0, 17: 707.3, 18: 708.7, 19: 710.0,
    20: 711.3, 21: 712.7, 22: 714.0, 23: 715.3, 24: 716.7, 25: 718.0,
    26: 946.0, 27: 947.3, 28: 948.7, 29: 950.0, 30: 951.3, 31: 952.7,
    32: 954.0, 33: 955.3, 34: 956.7, 35: 958.0, 36: 959.3, 37: 960.7,
    38: 962.0, 39: 963.3, 40: 964.7, 41: 966.0, 42: 967.3, 43: 968.7,
    44: 970.0, 45: 971.3, 46: 972.7, 47: 974.0, 48: 975.3, 49: 976.7,
    50: 978.0, 51: 707.0, 52: 708.8, 53: 710.5, 54: 712.3, 55: 714.0,
    56: 715.8, 57: 717.6, 58: 719.3, 59: 721.1, 60: 722.9, 61: 724.6,
    62: 726.4, 63: 728.1, 64: 729.9, 65: 731.7, 66: 733.4, 67: 735.2,
    68: 737.0, 69: 738.7, 70: 740.5, 71: 742.2, 72: 744.0, 73: 967.0,
    74: 968.7, 75: 970.3, 76: 972.0, 77: 973.7, 78: 975.3, 79: 977.0,
    80: 978.7, 81: 980.3, 82: 982.0, 83: 983.7, 84: 985.3, 85: 987.0,
    86: 988.7, 87: 990.3, 88: 992.0, 89: 993.7, 90: 995.3, 91: 997.0,
    92: 998.7, 93: 1000.3, 94: 1002.0,
}

# Same brightness-model parameters as step4_brightness_20deg.py
POWER_KW = 400.0
CONTINUOUS = False
OFFSET_DEG = 20

RE_KM = 6378.137  # Earth radius, matches step2's compute_sunlit()

# ─────────────────────────────────────────────────────────────────────────
# 532 nm -> LSST band correction
# ─────────────────────────────────────────────────────────────────────────
# calculate_brightness() (and therefore RealBrightnessBackend.ab_magnitudes)
# returns AB magnitude at a single ~532 nm passband regardless of which
# LSST filter is actually being compared against -- exactly the same gap
# step4b_band_correction.py fixed for the step4 pixel-loss pipeline. Reused
# here verbatim (solar colours from Willmer 2018 / SMTN-002) so a
# satellite's magnitude is corrected to the SAME band as whatever natural
# sky baseline it's being measured against, instead of silently comparing
# a 532 nm number to an r/g/whatever-band mag/arcsec^2 value.
#
# --band-correction controls when this gets applied (see CLI help):
#   auto (default) -- only in --sky-baseline real/opsim, where the natural
#                      baseline is genuinely in a specific LSST band.
#   on             -- always, including --sky-baseline fixed. Your
#                      responsibility to also pick a band-appropriate
#                      --natural-sky-mag-arcsec2 in that case -- the fixed
#                      21.8 default is only "roughly V-band", not any
#                      specific LSST filter.
#   off            -- never (original, uncorrected 532 nm behavior).

SOLAR_COLOR = {
    'u': +1.428,
    'g': +0.245,
    'r': -0.210,
    'i': -0.322,
    'z': -0.357,
    'y': -0.371,
}


def base_band(filter_value) -> str:
    """Extracts the base LSST band letter from a (possibly suffixed)
    value, e.g. "r_57" -> "r" -- identical to step4b_band_correction.py's
    base_band(), reused here in case --band is ever passed as something
    other than a bare letter."""
    return str(filter_value).strip().split('_')[0]


def band_correct_ab_mag(ab_mag_532nm, band):
    """
    ab_mag_532nm + SOLAR_COLOR[band] -> AB magnitude in `band` instead of
    532 nm. Raises with a clear message (rather than silently returning
    NaN like step4b tolerates per-row) if `band` isn't a recognized LSST
    filter, since here it's one global setting for the whole run and a
    typo should fail loudly, not quietly NaN out every satellite.
    """
    b = base_band(band)
    if b not in SOLAR_COLOR:
        raise KeyError(
            f"--band {band!r} (base band {b!r}) has no solar-color entry "
            f"in SOLAR_COLOR -- expected one of {sorted(SOLAR_COLOR)}."
        )
    return np.asarray(ab_mag_532nm, dtype=np.float64) + SOLAR_COLOR[b]


# ═════════════════════════════════════════════════════════════════════════
# Orbital mechanics -- IDENTICAL formulas to step1.py / step2.py, so results
# are exactly consistent with your existing pipeline's satellite positions.
# ═════════════════════════════════════════════════════════════════════════

def mjd_to_jd_split(mjd):
    jd_full = mjd + 2400000.5
    jd1 = math.floor(jd_full)
    jd2 = jd_full - jd1
    return jd1, jd2


def gmst_rad(jd_ut1):
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = (67310.54841
                + (876600.0 * 3600.0 + 8640184.812866) * T
                + 0.093104 * T ** 2
                - 6.2e-6 * T ** 3)
    return math.radians(gmst_sec % 86400.0 / 240.0)


def teme_to_gcrs(pos_teme, jd_ut1):
    """pos_teme: (3, N) array. Identical to step2.teme_to_gcrs()."""
    theta = gmst_rad(jd_ut1)
    ct, st = math.cos(theta), math.sin(theta)
    return np.vstack([
        ct * pos_teme[0] + st * pos_teme[1],
        -st * pos_teme[0] + ct * pos_teme[1],
        pos_teme[2],
    ])


def gcrs_to_radec(pos_gcrs):
    """Identical to step2.gcrs_to_radec()."""
    x, y, z = pos_gcrs
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ra = (np.degrees(np.arctan2(y, x)) % 360.0).astype(np.float32)
    dec = np.degrees(np.arcsin(np.clip(z / r, -1.0, 1.0))).astype(np.float32)
    return ra, dec


def compute_sunlit(pos_gcrs, jd_full):
    """
    Exact cylindrical Earth-shadow model, identical to step2.compute_sunlit().
    pos_gcrs: (3, N) array in km. jd_full: scalar or (N,) array.
    """
    n_jd = jd_full - 2451545.0
    L = np.radians((280.460 + 0.9856474 * n_jd) % 360)
    g = np.radians((357.528 + 0.9856003 * n_jd) % 360)
    lam = L + np.radians(1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 4e-7 * n_jd)
    sun = np.vstack([np.cos(lam) * np.ones_like(pos_gcrs[0]),
                      (np.cos(eps) * np.sin(lam)) * np.ones_like(pos_gcrs[0]),
                      (np.sin(eps) * np.sin(lam)) * np.ones_like(pos_gcrs[0])])
    proj = -np.einsum('it,it->t', sun, pos_gcrs)
    r2 = np.einsum('it,it->t', pos_gcrs, pos_gcrs)
    return (proj < 0) | (r2 - proj ** 2 > RE_KM ** 2)


# ═════════════════════════════════════════════════════════════════════════
# Brightness backend -- swap MockBrightnessBackend for the real one in your
# environment. Interface matches data_center_20deg.calculate_brightness()
# exactly, so RealBrightnessBackend is a near-trivial wrapper.
# ═════════════════════════════════════════════════════════════════════════

class RealBrightnessBackend:
    """Wraps your actual data_center_20deg.calculate_brightness(). Use this
    when running on your real cluster/environment where lumos/starlink/
    analysis are installed."""

    def __init__(self):
        import data_center_20deg as data_center
        self._data_center = data_center

    def ab_magnitudes(self, sat_height_m, sat_alt_deg, sat_az_deg,
                       sun_alt_deg, sun_az_deg):
        result = self._data_center.calculate_brightness(
            sat_height=sat_height_m,
            sat_altitude=sat_alt_deg,
            sat_azimuth=sat_az_deg,
            sun_altitude=sun_alt_deg,
            sun_azimuth=sun_az_deg,
            power_kw=POWER_KW,
            continuous=CONTINUOUS,
            offset_deg=OFFSET_DEG,
        )
        return np.asarray(result["ab_magnitude"])


class MockBrightnessBackend:
    """
    STAND-IN used only to validate the rest of this pipeline (propagation,
    eclipse, Alt/Az, flux-summing) without the real lumos/starlink/analysis
    packages. NOT physically calibrated -- do not use for real results.

    Uses a simple inverse-square + phase-angle-ish falloff so brightness
    varies plausibly with elevation, purely to exercise the array-shapes
    and aggregation math end-to-end.
    """

    def ab_magnitudes(self, sat_height_m, sat_alt_deg, sat_az_deg,
                       sun_alt_deg, sun_az_deg):
        sat_alt_deg = np.atleast_1d(sat_alt_deg).astype(float)
        # Fainter near the horizon (larger range), brighter near zenith --
        # just enough structure to sanity-check the pipeline, not real physics.
        range_km = (sat_height_m / 1000.0) / np.maximum(np.sin(np.radians(
            np.clip(sat_alt_deg, 5, 90))), 0.1)
        base_mag = 4.0 + 5.0 * np.log10(range_km / 550.0)
        return base_mag


# ═════════════════════════════════════════════════════════════════════════
# Real per-timestep, per-sky-position natural brightness baseline
# ─────────────────────────────────────────────────────────────────────────
# Replaces the single fixed --natural-sky-mag-arcsec2 scalar with Rubin's
# own operational sky brightness forecast (rubin_scheduler.skybrightness_pre
# .SkyModelPre) -- the same pre-calculated healpix-map engine described in
# RTN-012, "Approximating Pre-calculated Sky Brightness with Zernike
# Coefficients" (https://rtn-012.lsst.io/), built from ESO skycalc and used
# operationally by the Rubin scheduler itself. This captures real
# night-to-night and within-night variation (moonrise/set, moon phase,
# twilight) instead of one flat mag/arcsec^2 everywhere, all night, every
# night.
#
# mu_natural at each timestep is a single SCALAR: the mean over every
# currently-unmasked (Moon/zenith/high-airmass-excluded), above-cutoff
# healpix pixel SkyModelPre reports for the requested band. That scalar
# plugs into the exact same whole-sky formula --sky-baseline fixed/opsim
# use, with every visible+sunlit satellite (down to elevation_cutoff_deg,
# same criterion as n_visible) contributing to the comparison -- NOT just
# whichever satellites happen to land in SkyModelPre's own valid pixels.
# An earlier version of this code bin satellite flux onto a healpix grid
# and computed delta_mag per-pixel, which sounds more rigorous but had a
# real, silent bug: SkyModelPre masks anything below ~24 deg altitude
# (airmass>2.5) regardless of our own elevation_cutoff_deg, so satellites
# counted in n_visible at low altitude contributed NOTHING to delta_mag --
# producing exactly-zero delta_mag for long stretches of real nights
# despite hundreds of thousands of "visible" satellites. The scalar
# approach here doesn't have that blind spot, at the cost of not being
# able to report a spatial hotspot (no more peak_pixel_delta_mag).
#
# CAVEATS -- read before trusting numbers from --sky-baseline real:
#   1. BAND MATCH: SkyModelPre reports per LSST filter (u/g/r/i/z/y). Your
#      calculate_brightness()/data_center_20deg model returns AB magnitudes
#      in whatever band it's calibrated in (roughly V-band, per the
#      original 21.8 mag/arcsec^2 default baseline). Pick --band to match
#      that as closely as you can; there is no automatic cross-band
#      conversion here. If the bands don't match, delta_mag/pct_increase
#      are still internally consistent but not an apples-to-apples number.
#   2. API SURFACE: SkyModelPre's method name has changed across the
#      lsst.sims -> rubin_sim -> rubin_scheduler package history. This is
#      written against the current rubin_scheduler.skybrightness_pre API
#      (SkyModelPre.return_mags(mjd, indx=...)); _query_sky_mags() below
#      falls back to the older returnMags() name and raises a clear error
#      pointing you at help(sky_model) if neither works on your installed
#      version -- verify against your actual environment before trusting
#      the numbers, the same way the original script flags
#      calculate_brightness() itself as unverified against your real env.
#   3. DATA FILES: SkyModelPre needs the pre-computed skybrightness_pre
#      data product on disk (e.g. `scheduler_download_data --dirs
#      skybrightness_pre`, or set RUBIN_SIM_DATA_DIR / pass
#      --rubin-sim-data-path). Nothing here downloads it for you.
#   4. RESOLUTION: mu_natural is averaged over a healpix grid
#      (--healpix-nside, default 32, matching SkyModelPre's own native
#      resolution). RealSkyBaseline auto-detects and self-corrects the
#      grid to whatever nside SkyModelPre actually uses on your installed
#      version, in case a future release changes it -- see
#      _verify_native_nside().
# ═════════════════════════════════════════════════════════════════════════

DEFAULT_HEALPIX_NSIDE = 32  # matches SkyModelPre's own native nside
                             # (confirmed via check_native_nside.py against
                             # a real rubin_scheduler 4.5.0 install -- was
                             # 16 before, which silently mismatched
                             # SkyModelPre's actual grid and caused
                             # near-total, geometry-uncorrelated masking).
                             # RealSkyBaseline now auto-detects and
                             # self-corrects if this is ever wrong again
                             # (see _verify_native_nside), but starting
                             # from the right value avoids that extra
                             # check/warning on every normal run.


def _query_sky_mags(sky_model, mjd, indx, band):
    """
    Calls SkyModelPre for natural sky brightness at a set of healpix ids.
    Handles the return_mags/returnMags naming split across package
    versions (see the module-level caveat above). Returns a 1-D array of
    mag/arcsec^2, same length/order as indx, for the requested band.

    Explicitly requests badval=np.nan: SkyModelPre masks certain pixels
    (near the Moon, near zenith, high airmass, near bright planets) by
    filling them with a sentinel value -- by default a huge number
    (-1.6375e30), NOT NaN. Left unhandled, that sentinel silently
    overflows to +inf the moment anything does 10**(-0.4*mag) on it,
    which then poisons a whole-dome mean to inf, then to NaN the moment
    two such means get subtracted (delta_mag = mu_natural - mu_combined
    = -inf - (-inf) = NaN) -- even from just ONE masked pixel out of
    thousands. Requesting NaN directly lets us filter masked pixels out
    explicitly instead of discovering them this way.
    """
    kwargs = dict(badval=np.nan)
    if hasattr(sky_model, "return_mags"):
        mags = sky_model.return_mags(mjd, indx=indx, **kwargs)
    elif hasattr(sky_model, "returnMags"):
        mags = sky_model.returnMags(mjd, indx, **kwargs)
    else:
        raise AttributeError(
            "SkyModelPre has neither return_mags() nor returnMags() -- "
            "run `help(sky_model)` in your environment to find the "
            "correct method for your installed rubin_scheduler/rubin_sim "
            "version and update _query_sky_mags() to match."
        )
    if band not in mags:
        raise KeyError(
            f"Band {band!r} not in SkyModelPre output (got keys "
            f"{list(mags.keys())}). Check --band."
        )
    return np.asarray(mags[band], dtype=np.float64)


class RealSkyBaseline:
    """
    Wraps rubin_scheduler.skybrightness_pre.SkyModelPre to provide, at any
    timestep in the run, the natural (satellite-free) sky brightness at
    every currently-visible healpix pixel above Rubin, in the requested
    LSST filter. Satellite positions are binned onto the same grid so the
    two can be combined pixel-by-pixel. See the module-level caveats above
    -- this needs the real rubin_scheduler package + data files, and its
    exact API has not been verified against your installed version.
    """

    def __init__(self, band, nside=DEFAULT_HEALPIX_NSIDE, data_path=None):
        try:
            from rubin_scheduler.skybrightness_pre import SkyModelPre
        except ImportError as e:
            raise ImportError(
                "--sky-baseline real requires the `rubin_scheduler` "
                "package (`pip install rubin_scheduler`) plus its "
                "skybrightness_pre data product on disk (see "
                "`rs_download_data --dirs skybrightness_pre` or set "
                "RUBIN_SIM_DATA_DIR / --rubin-sim-data-path). Use "
                "--sky-baseline fixed if you don't have these available."
            ) from e

        import healpy as hp

        self.band = band
        self.hp = hp
        self.sky_model = SkyModelPre(data_path=data_path)
        self._requested_nside = nside
        self._nside_verified = False
        self._build_grid(nside)

    def _build_grid(self, nside):
        """(Re)builds the pixel grid at the given nside. Split out from
        __init__ so _verify_native_nside() can rebuild it if the
        requested nside doesn't match SkyModelPre's actual data."""
        hp = self.hp
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(self.npix))
        self.pix_ra_deg = np.degrees(phi)
        self.pix_dec_deg = 90.0 - np.degrees(theta)
        self.pixel_area_arcsec2 = hp.nside2pixarea(nside) * (180.0 * 3600.0 / math.pi) ** 2

    def _verify_native_nside(self, mjd):
        """
        SkyModelPre.return_mags()'s `indx` parameter is "the healpix ID"
        -- in SkyModelPre's OWN native nside, not whatever nside we
        happen to have built our grid at. Passing indices built for the
        wrong nside doesn't error (they're still valid array indices,
        just pointing at the wrong sky positions), so this bug is silent
        and shows up as pervasive, geometrically-uncorrelated "masking"
        -- exactly what happened with the original nside=16 default
        against a native nside=32 dataset: ~97% of pixels came back
        masked, and that 97% correlated with none of airmass, zenith
        distance, or Moon separation, because the indices weren't
        actually pointing at the alt/az we thought they were.

        Runs once (lazily, on the first real query, since we need a
        valid mjd and don't have one yet at __init__ time), and rebuilds
        the grid at the correct nside automatically if it doesn't match
        -- so this class self-corrects rather than silently producing
        wrong results if a future rubin_scheduler version ships a
        different native resolution.
        """
        if self._nside_verified:
            return
        full_sky = _query_sky_mags(self.sky_model, mjd, None, self.band)
        native_nside = self.hp.npix2nside(len(full_sky))
        if native_nside != self.nside:
            print(f"[WARN] RealSkyBaseline: requested nside={self._requested_nside} "
                  f"does not match SkyModelPre's actual native nside="
                  f"{native_nside} -- rebuilding the pixel grid at "
                  f"{native_nside} to match. (Indices for the wrong "
                  f"nside are still valid array indices, just pointing "
                  f"at the wrong sky positions -- this is what caused "
                  f"near-total, geometry-uncorrelated 'masking' before "
                  f"this check existed.)")
            self._build_grid(native_nside)
        self._nside_verified = True

    def radec_to_pix(self, ra_deg, dec_deg):
        """Bins arbitrary RA/Dec (e.g. satellite positions) onto this
        object's healpix grid. Ring ordering, matching pix2ang above."""
        if len(ra_deg) == 0:
            return np.array([], dtype=np.int64)
        theta = np.radians(90.0 - dec_deg)
        phi = np.radians(ra_deg)
        return self.hp.ang2pix(self.nside, theta, phi)

    def visible_pixels(self, mjd, location, elevation_cutoff_deg):
        """Returns (pix_ids, natural_mag) for every pixel above the
        elevation cutoff at this mjd, in this object's band, EXCLUDING
        any SkyModelPre masks (Moon/zenith/airmass/bright-planet
        exclusions -- see _query_sky_mags). Without this filter, masked
        pixels poison the whole-dome mean (see _query_sky_mags's
        docstring); with it, they're just correctly treated as "no
        natural-sky data here" and dropped from the average, the same
        way a real observer would simply not point there."""
        self._verify_native_nside(mjd)
        alt, _ = radec_to_altaz(self.pix_ra_deg, self.pix_dec_deg, mjd, location)
        vis = np.where(alt >= elevation_cutoff_deg)[0]
        if len(vis) == 0:
            return vis, np.array([])
        mags = _query_sky_mags(self.sky_model, mjd, vis, self.band)
        finite = np.isfinite(mags)
        n_masked = int((~finite).sum())
        if n_masked > 0:
            frac = n_masked / len(vis)
            if frac > 0.5:
                print(f"[WARN] mjd={mjd:.5f}: SkyModelPre masked "
                      f"{n_masked}/{len(vis)} ({frac:.0%}) of the visible "
                      f"dome for band {self.band!r} (Moon/zenith/airmass/"
                      f"planet exclusion) -- mu_natural for this "
                      f"timestep is being averaged over a much smaller "
                      f"unmasked remainder than usual.")
        return vis[finite], mags[finite]


# ═════════════════════════════════════════════════════════════════════════
# Real, already-simulated per-visit sky brightness pulled straight out of a
# Rubin/LSST opsim run (e.g. baseline_v5.3.0_10yrs.db) -- an alternative to
# RealSkyBaseline above that needs NO extra package or downloaded data
# product beyond the .db file itself, since the opsim run already contains
# the scheduler's own forecast sky brightness for every visit it made.
#
# TRADE-OFF vs RealSkyBaseline: this gives a real, band-matched
# mu_natural(t) TIME series for the requested night (built by interpolating
# the actual visits' skyBrightness values), but NOT a full-sky spatial map
# at each instant -- an opsim run only points at a scattering of individual
# fields at any given moment, not densely enough across the whole sky to
# reconstruct a moon-centered brightness map the way SkyModelPre can. So
# --sky-baseline opsim plugs into the ORIGINAL whole-sky-scalar formula
# (like --sky-baseline fixed) but with a time-varying mu_natural(t) instead
# of one flat constant, rather than the pixel-by-pixel binning
# --sky-baseline real does.
#
# CAVEATS:
#   - Your db's `filter` column is the physical filter+serial number (e.g.
#     "u_24"); `band` is the clean ugrizy value -- this matches on `band`.
#     Confirm your own db has both before assuming this.
#   - Some bands (esp. u) are used sparingly by the scheduler -- a
#     specific night can easily have zero visits in the requested band.
#     When that happens this falls back to the nearest real visits in
#     that band ANYWHERE in the 10-year run and prints a clear warning;
#     check .used_fallback / .fallback_gap_days rather than assuming the
#     curve came from the requested night.
# ═════════════════════════════════════════════════════════════════════════

class OpsimSkyBaseline:
    """Builds a real, band-specific mu_natural(mjd) curve for one night by
    pulling every actual opsim visit in that band during the requested
    night's MJD window and linearly interpolating skyBrightness between
    them. See the module-level caveats above."""

    def __init__(self, db_path, band, night_mjd_start, night_mjd_end, verbose=False):
        if not os.path.isfile(db_path):
            raise FileNotFoundError(f"opsim database not found: {db_path!r}")

        self.band = band
        self.used_fallback = False
        self.fallback_gap_days = None

        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = con.cursor()

        rows = cur.execute(
            'SELECT observationStartMJD, skyBrightness FROM observations '
            'WHERE band = ? AND observationStartMJD BETWEEN ? AND ? '
            'ORDER BY observationStartMJD',
            (band, night_mjd_start, night_mjd_end),
        ).fetchall()

        if not rows:
            self.used_fallback = True
            mid_mjd = 0.5 * (night_mjd_start + night_mjd_end)
            rows = cur.execute(
                'SELECT observationStartMJD, skyBrightness FROM observations '
                'WHERE band = ? ORDER BY ABS(observationStartMJD - ?) LIMIT 50',
                (band, mid_mjd),
            ).fetchall()
            if not rows:
                con.close()
                raise ValueError(
                    f"No visits in band {band!r} found anywhere in "
                    f"{db_path!r}. Check --band against this db's actual "
                    f"`band` column values."
                )
            rows = sorted(rows, key=lambda r: r[0])
            self.fallback_gap_days = min(abs(r[0] - mid_mjd) for r in rows)
            print(f"[WARN] No {band!r}-band visits found on the requested "
                  f"night -- falling back to the nearest {len(rows)} "
                  f"{band!r}-band visits anywhere in the survey "
                  f"({self.fallback_gap_days:.1f} days away). "
                  f"mu_natural(t) reflects THAT data, not this specific "
                  f"night's forecast.")

        con.close()

        self.mjd_arr = np.array([r[0] for r in rows], dtype=np.float64)
        self.mag_arr = np.array([r[1] for r in rows], dtype=np.float64)

        # Always print this (not gated behind --verbose) -- it's the one
        # diagnostic that actually explains a flat-looking mu_natural(t):
        # np.interp does NOT extrapolate, so if the requested night's MJD
        # window doesn't overlap the range of points found here (whether
        # from an exact night+band match or the fallback above), every
        # single query for that night clamps to one boundary value and
        # the whole curve comes out flat, even though real varying data
        # exists elsewhere in the survey.
        print(f"[INFO] OpsimSkyBaseline: {len(self.mjd_arr)} real "
              f"{band!r}-band visits, mjd {self.mjd_arr.min():.5f}-"
              f"{self.mjd_arr.max():.5f}, skyBrightness "
              f"{self.mag_arr.min():.3f}-{self.mag_arr.max():.3f}")

        data_start, data_end = self.mjd_arr.min(), self.mjd_arr.max()
        no_overlap = (data_end < night_mjd_start) or (data_start > night_mjd_end)
        if no_overlap or len(self.mjd_arr) == 1:
            reason = ("only 1 matching visit exists" if len(self.mjd_arr) == 1
                       else "the requested night's window doesn't overlap "
                            "this data's mjd range at all")
            print(f"[WARN] mu_natural(t) will be FLAT (a single clamped "
                  f"value) for this ENTIRE night -- {reason}. This is "
                  f"expected behavior for np.interp with no extrapolation, "
                  f"not a bug, but it means this run isn't capturing any "
                  f"real within-night sky-brightness variation for "
                  f"{band!r}. Consider --sky-baseline real (SkyModelPre) "
                  f"if you need a genuine forecast for a band this "
                  f"sparsely observed.")

    def mu_natural(self, mjd):
        """Linearly interpolates the real observed/forecast sky brightness
        to an arbitrary mjd; np.interp clamps (no extrapolation) at the
        ends of whatever data was actually pulled, rather than guessing
        beyond it."""
        return float(np.interp(mjd, self.mjd_arr, self.mag_arr))


def _default_opsim_db_path():
    """Mirrors inspect_opsim_db.py's convention: look for
    baseline_v5.3.0_10yrs.db right next to this script if --opsim-db isn't
    given explicitly."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        here = os.getcwd()
    return os.path.join(here, "baseline_v5.3.0_10yrs.db")


# ═════════════════════════════════════════════════════════════════════════
# TLE loading + propagation
# ═════════════════════════════════════════════════════════════════════════

def load_shell_tles(shell_id, date_str, tle_dir):
    """Loads the FULL constellation TLE set for one shell/date (not the
    rep-sat subset step1's own analysis uses)."""
    path = os.path.join(tle_dir, f"tles_shell{shell_id:03d}_{date_str}.pkl")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        cached = pickle.load(f)
    return cached["tle_lines"] if isinstance(cached, dict) else cached


def propagate_shell(tle_lines, jd1, jd2):
    """
    Propagates every satellite in a shell to ONE timestep.

    FIX: uses sgp4.api.SatrecArray (the package's own supported
    multi-satellite batch API) instead of looping and calling
    .sgp4_array() on freshly-created individual Satrec objects one at a
    time. The per-object loop approach was found to be genuinely
    non-deterministic and frequently returned all-NaN positions for every
    satellite after the first in a shell -- confirmed by re-running the
    identical (TLE, jd1, jd2) through .sgp4_array() twice and getting two
    different answers. SatrecArray does not exhibit this: verified
    zero-NaN and bit-identical results across repeated calls on the same
    input during testing.

    Returns (ra_deg, dec_deg, sunlit_mask) arrays, one entry per satellite
    that parsed successfully (failed TLEs are dropped silently, same
    tolerance as step1/step2).
    """
    from sgp4.api import Satrec, SatrecArray

    sats = []
    for tle in tle_lines:
        parts = tle.strip().split("\n")
        if len(parts) != 3:
            continue
        _, l1, l2 = parts
        try:
            sats.append(Satrec.twoline2rv(l1, l2))
        except Exception:
            continue

    if not sats:
        return (np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=bool))

    sat_array = SatrecArray(sats)
    e, r, _ = sat_array.sgp4(np.array([jd1]), np.array([jd2]))
    # e, r shapes: (n_sats, 1), (n_sats, 1, 3)
    #
    # IMPORTANT: e==0 ("no error") is NOT sufficient to guarantee a valid
    # position -- confirmed by direct testing that specific (satellite,
    # timestep) combinations return e==0 while r is NaN. Must explicitly
    # check for NaN as an independent failure condition, same defensive
    # "drop and continue" tolerance step1/step2 use for actual error codes.
    r0 = r[:, 0, :]
    ok = (e[:, 0] == 0) & ~np.isnan(r0).any(axis=1)
    if not ok.any():
        return (np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=bool))

    pos_teme = r0[ok, :].T  # (3, n_ok)
    pos_gcrs = teme_to_gcrs(pos_teme, jd1 + jd2)
    ra, dec = gcrs_to_radec(pos_gcrs)
    sunlit = compute_sunlit(pos_gcrs, np.full(pos_gcrs.shape[1], jd1 + jd2))

    return (ra.astype(np.float64), dec.astype(np.float64), sunlit)


def radec_to_altaz(ra_deg, dec_deg, mjd_utc, location):
    """Identical astropy transform to step3a.compute_azel()."""
    import astropy.time
    import astropy.units as u
    from astropy.coordinates import SkyCoord, AltAz, ICRS
    import warnings

    if len(ra_deg) == 0:
        return np.array([]), np.array([])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs_time = astropy.time.Time(mjd_utc, format="mjd", scale="utc")
        frame = AltAz(obstime=obs_time, location=location)
        sky = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame=ICRS())
        altaz = sky.transform_to(frame)
    return altaz.alt.deg, altaz.az.deg


def sun_altaz(mjd_utc, location):
    """Identical to step3b.compute_sun_azel() for one timestamp."""
    import astropy.time
    from astropy.coordinates import AltAz, get_sun
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obs_time = astropy.time.Time(mjd_utc, format="mjd", scale="utc")
        frame = AltAz(obstime=obs_time, location=location)
        sun_coord = get_sun(obs_time)
        sun_aa = sun_coord.transform_to(frame)
    return float(sun_aa.alt.deg), float(sun_aa.az.deg)


# ═════════════════════════════════════════════════════════════════════════
# One-timestep aggregate
# ═════════════════════════════════════════════════════════════════════════

def _gather_visible_satellites(mjd, tle_cache, location, elevation_cutoff_deg,
                                backend, band=None, apply_band_correction=False,
                                verbose=False):
    """
    Shared by both baseline modes below: propagates every shell, keeps
    sunlit + above-cutoff satellites, and returns their equatorial RA/Dec
    (deg) and AB magnitudes as flat per-satellite arrays (NOT yet summed
    or binned), plus the Sun's Alt/Az. This is exactly the same
    propagate -> sunlit-filter -> alt/az-filter -> calculate_brightness
    sequence the original compute_one_timestep used; it's just factored
    out so the fixed-baseline and real-sky-baseline paths can't drift
    apart from each other.

    apply_band_correction=True adds SOLAR_COLOR[band] to every magnitude
    right after calculate_brightness() returns it (532 nm -> `band`), so
    everything downstream -- flux summing, healpix binning, whole-dome
    aggregation -- already operates on band-corrected magnitudes.
    """
    jd1, jd2 = mjd_to_jd_split(mjd)
    s_alt, s_az = sun_altaz(mjd, location)

    all_ra, all_dec, all_mag = [], [], []

    for shell_id, tle_lines in tle_cache.items():
        if tle_lines is None:
            continue
        ra, dec, sunlit = propagate_shell(tle_lines, jd1, jd2)
        if len(ra) == 0:
            continue

        sunlit_idx = np.where(sunlit)[0]
        if len(sunlit_idx) == 0:
            continue

        alt, az = radec_to_altaz(ra[sunlit_idx], dec[sunlit_idx], mjd, location)
        visible_mask = alt >= elevation_cutoff_deg
        n_vis = int(visible_mask.sum())
        if n_vis == 0:
            continue

        alt_km = SHELL_ALT_KM[shell_id]
        ab_mags = backend.ab_magnitudes(
            sat_height_m=alt_km * 1000.0,
            sat_alt_deg=alt[visible_mask],
            sat_az_deg=az[visible_mask],
            sun_alt_deg=s_alt,
            sun_az_deg=s_az,
        )
        ab_mags = np.asarray(ab_mags, dtype=np.float64)
        if apply_band_correction:
            ab_mags = band_correct_ab_mag(ab_mags, band)
        all_ra.append(ra[sunlit_idx][visible_mask])
        all_dec.append(dec[sunlit_idx][visible_mask])
        all_mag.append(ab_mags)

        if verbose:
            print(f"    shell {shell_id:3d}: {len(ra):5d} sunlit, "
                  f"{n_vis:5d} above {elevation_cutoff_deg}deg elevation")

    if all_ra:
        ra_out = np.concatenate(all_ra)
        dec_out = np.concatenate(all_dec)
        mag_out = np.concatenate(all_mag)
    else:
        ra_out = np.array([])
        dec_out = np.array([])
        mag_out = np.array([])

    return ra_out, dec_out, mag_out, s_alt, s_az


def compute_one_timestep(mjd, tle_cache, location, backend,
                          elevation_cutoff_deg, band=None,
                          apply_band_correction=False, verbose=False):
    """
    FIXED-BASELINE mode (original behavior when apply_band_correction is
    False, which was always true before this parameter existed): returns
    dict with n_visible, flux_total, sun_alt_deg, sun_az_deg, summed
    across every visible+sunlit satellite regardless of where on the sky
    it is.
    """
    ra, dec, ab_mags, s_alt, s_az = _gather_visible_satellites(
        mjd, tle_cache, location, elevation_cutoff_deg, backend,
        band=band, apply_band_correction=apply_band_correction, verbose=verbose)
    flux_total = float(np.sum(10.0 ** (-0.4 * ab_mags))) if len(ab_mags) else 0.0
    return {
        "mjd": mjd, "sun_alt_deg": s_alt, "sun_az_deg": s_az,
        "n_visible": len(ab_mags), "flux_total": flux_total,
    }


def compute_one_timestep_real_sky(mjd, tle_cache, location, backend,
                                   elevation_cutoff_deg, sky_baseline,
                                   apply_band_correction=True, verbose=False):
    """
    REAL-SKY-BASELINE mode (simplified): pulls a real, band-matched,
    per-timestep mu_natural SCALAR from SkyModelPre -- averaged over
    every currently-unmasked (valid) pixel above the elevation cutoff --
    then plugs it into the exact SAME whole-sky-scalar formula
    --sky-baseline fixed/opsim use, with EVERY visible+sunlit satellite
    down to elevation_cutoff_deg (same criterion as n_visible)
    contributing to flux_total.

    This replaces an earlier per-pixel-binning version of this function
    that only counted satellite flux landing in one of SkyModelPre's own
    unmasked pixels. Since SkyModelPre masks airmass>2.5 (~alt<24 deg)
    regardless of our own elevation_cutoff_deg (commonly 0.0), that
    version silently dropped any contribution from satellites below
    ~24 deg altitude -- even though they were still being counted in
    n_visible -- which produced exactly-zero delta_mag for long
    stretches of real nights with hundreds of thousands of "visible"
    satellites. Not because the true whole-sky effect was zero, but
    because none of the counted satellites happened to land in the
    fraction of sky SkyModelPre would let us compare against.

    Trade-off: this drops peak_pixel_delta_mag (spatial hotspot
    detection) -- there's no longer a per-pixel combined map to search
    for a worst spot in, since satellite flux is no longer binned by
    position. Worth it: every satellite that counts toward n_visible now
    also counts toward delta_mag, with no altitude blind spot.
    """
    vis_pix, natural_mag = sky_baseline.visible_pixels(mjd, location, elevation_cutoff_deg)
    mu_natural = (-2.5 * math.log10(np.mean(10.0 ** (-0.4 * natural_mag)))
                  if len(vis_pix) else float("nan"))

    ra, dec, ab_mags, s_alt, s_az = _gather_visible_satellites(
        mjd, tle_cache, location, elevation_cutoff_deg, backend,
        band=sky_baseline.band, apply_band_correction=apply_band_correction,
        verbose=verbose)
    flux_total = float(np.sum(10.0 ** (-0.4 * ab_mags))) if len(ab_mags) else 0.0

    if not np.isfinite(mu_natural):
        return {
            "mjd": mjd, "sun_alt_deg": s_alt, "sun_az_deg": s_az,
            "n_visible": len(ab_mags), "flux_total": flux_total,
            "mu_natural": mu_natural, "mu_combined": float("nan"),
            "delta_mag": float("nan"), "pct_increase": float("nan"),
        }

    omega = solid_angle_arcsec2(elevation_cutoff_deg)
    f_natural = 10.0 ** (-0.4 * mu_natural)
    f_added_per_arcsec2 = flux_total / omega
    mu_combined = (-2.5 * math.log10(f_natural + f_added_per_arcsec2)
                   if (f_natural + f_added_per_arcsec2) > 0 else float("nan"))
    delta_mag = mu_natural - mu_combined
    pct_increase = 100.0 * f_added_per_arcsec2 / f_natural

    return {
        "mjd": mjd, "sun_alt_deg": s_alt, "sun_az_deg": s_az,
        "n_visible": len(ab_mags), "flux_total": flux_total,
        "mu_natural": mu_natural, "mu_combined": mu_combined,
        "delta_mag": delta_mag, "pct_increase": pct_increase,
    }


# ═════════════════════════════════════════════════════════════════════════
# Full-night driver
# ═════════════════════════════════════════════════════════════════════════

def solid_angle_arcsec2(elevation_cutoff_deg):
    """Solid angle of the sky dome above a given elevation, in arcsec^2.
    Omega = 2*pi*(1 - sin(elevation_cutoff)) steradians."""
    omega_sr = 2 * math.pi * (1 - math.sin(math.radians(elevation_cutoff_deg)))
    arcsec2_per_sr = (180.0 * 3600.0 / math.pi) ** 2
    return omega_sr * arcsec2_per_sr


def run_night(date_str, tle_dir, time_step_sec, elevation_cutoff_deg,
              natural_sky_mag_arcsec2, backend, night_sun_el_max_deg=-18.0,
              sky_baseline=None, opsim_db_path=None, opsim_band=None,
              apply_band_correction=False, verbose=False):
    """
    Three mutually exclusive baseline modes:
      sky_baseline=None, opsim_db_path=None (default) -> FIXED: exactly the
          original behavior, one constant natural_sky_mag_arcsec2 for the
          whole night/sky.
      sky_baseline=<RealSkyBaseline> -> REAL: per-timestep, per-position
          healpix map from SkyModelPre (natural_sky_mag_arcsec2 ignored).
      opsim_db_path=<path to opsim .db>, opsim_band=<'u'/'g'/.../'y'> ->
          OPSIM: a real, band-matched mu_natural(t) curve built from actual
          simulated visits in that db for this night (natural_sky_mag_arcsec2
          ignored). See OpsimSkyBaseline's docstring for caveats.

    apply_band_correction: whether satellite AB magnitudes get corrected
    from 532 nm to opsim_band/sky_baseline.band before being used (see
    band_correct_ab_mag()/SOLAR_COLOR). Only meaningful for the fixed
    baseline here -- REAL mode always applies its own correction using
    sky_baseline.band (see compute_one_timestep_real_sky).
    """
    import astropy.time
    import astropy.units as u
    from astropy.coordinates import EarthLocation

    location = EarthLocation(
        lat=RUBIN_LAT_DEG * u.deg, lon=RUBIN_LON_DEG * u.deg,
        height=RUBIN_ELEV_M * u.m,
    )

    # ── Load every shell's TLEs for this date once ─────────────────────
    print(f"[INFO] Loading TLEs for {len(SHELL_ALT_KM)} shells on {date_str} ...")
    tle_cache = {}
    n_missing = 0
    for shell_id in SHELL_ALT_KM:
        tle_lines = load_shell_tles(shell_id, date_str, tle_dir)
        if tle_lines is None:
            n_missing += 1
        tle_cache[shell_id] = tle_lines
    n_found = len(SHELL_ALT_KM) - n_missing
    print(f"[INFO] Found {n_found}/{len(SHELL_ALT_KM)} shell TLE files "
          f"(missing: {n_missing})")
    if n_found == 0:
        raise FileNotFoundError(
            f"No tles_shell*.pkl files found in {tle_dir!r} for {date_str}. "
            f"Check --tle-dir and that step1 has been run for this date."
        )

    # ── Find the astronomical-night window for this date ───────────────
    date = pd.Timestamp(date_str)
    mjd_noon = astropy.time.Time(date_str, format="iso", scale="utc").mjd + 0.5
    scan_mjds = np.linspace(mjd_noon, mjd_noon + 1.0, 24 * 12)  # 5-min scan
    print("[INFO] Scanning for astronomical-night window ...")
    sun_els = []
    for m in scan_mjds:
        salt, _ = sun_altaz(m, location)
        sun_els.append(salt)
    sun_els = np.array(sun_els)
    night_mask = sun_els <= night_sun_el_max_deg
    if not night_mask.any():
        raise ValueError(f"No astronomical night found on {date_str} "
                          f"(sun never drops below {night_sun_el_max_deg} deg)")
    night_start = scan_mjds[night_mask][0]
    night_end = scan_mjds[night_mask][-1]
    print(f"[INFO] Astronomical night: MJD {night_start:.5f} to {night_end:.5f} "
          f"({(night_end - night_start) * 24:.2f} hours)")

    time_step_days = time_step_sec / 86400.0
    n_steps = max(1, int((night_end - night_start) / time_step_days))
    timesteps = np.linspace(night_start, night_end, n_steps)
    print(f"[INFO] {n_steps} timestep(s) at {time_step_sec}s resolution")

    rows = []

    if sky_baseline is not None:
        # ── REAL per-timestep, per-position baseline ───────────────────
        print(f"[INFO] Using real per-timestep sky brightness from "
              f"SkyModelPre (band={sky_baseline.band}, "
              f"healpix nside={sky_baseline.nside}) ...")
        for i, mjd in enumerate(timesteps):
            r = compute_one_timestep_real_sky(
                mjd, tle_cache, location, backend, elevation_cutoff_deg,
                sky_baseline, apply_band_correction=apply_band_correction,
                verbose=verbose)
            rows.append(r)
            if (i + 1) % max(1, n_steps // 10) == 0 or i == n_steps - 1:
                print(f"  [{i+1}/{n_steps}] mjd={mjd:.5f}  "
                      f"n_visible={r['n_visible']:,}  "
                      f"mu_natural={r['mu_natural']:.3f}  "
                      f"delta_mag={r['delta_mag']:+.5f}  "
                      f"pct_increase={r['pct_increase']:.3f}%")
    else:
        # ── FIXED constant or OPSIM time-varying baseline ───────────────
        # Both share the original whole-sky-scalar formula; the only
        # difference is whether mu_nat is one constant for the whole
        # night or a real interpolated curve from actual opsim visits.
        opsim_baseline = None
        if opsim_db_path is not None:
            opsim_baseline = OpsimSkyBaseline(
                opsim_db_path, opsim_band, night_start, night_end, verbose=verbose)
            print(f"[INFO] Using real opsim sky brightness curve from "
                  f"{opsim_db_path} (band={opsim_band}) ...")

        omega = solid_angle_arcsec2(elevation_cutoff_deg)
        band_for_correction = opsim_band if opsim_band is not None else None

        for i, mjd in enumerate(timesteps):
            r = compute_one_timestep(mjd, tle_cache, location, backend,
                                      elevation_cutoff_deg,
                                      band=band_for_correction,
                                      apply_band_correction=apply_band_correction,
                                      verbose=verbose)
            mu_nat = (opsim_baseline.mu_natural(mjd) if opsim_baseline is not None
                      else natural_sky_mag_arcsec2)
            f_natural = 10.0 ** (-0.4 * mu_nat)
            f_added_per_arcsec2 = r["flux_total"] / omega
            mu_combined = (-2.5 * math.log10(f_natural + f_added_per_arcsec2)
                           if (f_natural + f_added_per_arcsec2) > 0 else float("nan"))
            delta_mag = mu_nat - mu_combined
            pct_increase = 100.0 * f_added_per_arcsec2 / f_natural
            rows.append({
                **r, "mu_natural": mu_nat, "delta_mag": delta_mag,
                "pct_increase": pct_increase, "mu_combined": mu_combined,
            })
            if (i + 1) % max(1, n_steps // 10) == 0 or i == n_steps - 1:
                print(f"  [{i+1}/{n_steps}] mjd={mjd:.5f}  "
                      f"n_visible={r['n_visible']:,}  "
                      f"mu_natural={mu_nat:.3f}  "
                      f"delta_mag={delta_mag:+.5f}  pct_increase={pct_increase:.3f}%")

    df = pd.DataFrame(rows)
    return df


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--tle-dir", default="streak_output_sgp4",
                     help="Directory containing tles_shell*.pkl (step1 output)")
    ap.add_argument("--time-step-sec", type=float, default=120.0)
    ap.add_argument("--elevation-cutoff-deg", type=float, default=0.0)
    ap.add_argument("--natural-sky-mag-arcsec2", type=float, default=21.8,
                     help="Only used with --sky-baseline fixed.")
    ap.add_argument("--night-sun-el-max-deg", type=float, default=-18.0)
    ap.add_argument("--sky-baseline", choices=["fixed", "real", "opsim"], default="fixed",
                     help="'fixed' (default): single --natural-sky-mag-arcsec2 "
                          "constant for the whole night and whole sky -- the "
                          "original behavior. 'real': pull the actual "
                          "per-timestep, per-band sky brightness forecast "
                          "from rubin_scheduler.skybrightness_pre.SkyModelPre "
                          "instead, varying with time (moon phase/position, "
                          "twilight) and sky position -- needs the real "
                          "rubin_scheduler package + its data files, see "
                          "RealSkyBaseline's docstring. 'opsim': pull a real, "
                          "band-matched mu_natural(t) curve straight out of "
                          "an opsim database (e.g. baseline_v5.3.0_10yrs.db) "
                          "via --opsim-db -- no extra package/data product "
                          "needed, but only varies in time, not sky position; "
                          "see OpsimSkyBaseline's docstring for caveats.")
    ap.add_argument("--band", default="r",
                     help="LSST filter/band (u/g/r/i/z/y) to query for "
                          "--sky-baseline real or opsim. Pick whichever is "
                          "closest to the band your brightness backend's AB "
                          "magnitudes are calibrated in -- there's no "
                          "automatic cross-band conversion here.")
    ap.add_argument("--healpix-nside", type=int, default=DEFAULT_HEALPIX_NSIDE,
                     help="Healpix resolution for --sky-baseline real: "
                          "binning satellite flux and querying natural sky "
                          "brightness. Higher = finer spatial resolution but "
                          "slower (every timestep re-transforms the whole "
                          "pixel grid to alt/az).")
    ap.add_argument("--rubin-sim-data-path", default=None,
                     help="Optional explicit path to the rubin_sim_data "
                          "directory (skybrightness_pre files), passed to "
                          "SkyModelPre(data_path=...). Defaults to whatever "
                          "RUBIN_SIM_DATA_DIR / your rubin_scheduler install "
                          "resolves to. Only used with --sky-baseline real.")
    ap.add_argument("--opsim-db", default=None,
                     help="Path to an opsim database (e.g. "
                          "baseline_v5.3.0_10yrs.db) for --sky-baseline "
                          "opsim. Defaults to a file of that name next to "
                          "this script, matching inspect_opsim_db.py's "
                          "convention.")
    ap.add_argument("--band-correction", choices=["auto", "on", "off"], default="auto",
                     help="Whether to correct satellite AB magnitudes from "
                          "532 nm to --band using solar-colour offsets "
                          "(SOLAR_COLOR, same table as "
                          "step4b_band_correction.py / Willmer 2018, "
                          "SMTN-002) before comparing against the natural "
                          "sky baseline. 'auto' (default): on for "
                          "--sky-baseline real/opsim (where the natural "
                          "baseline genuinely is in --band), off for fixed "
                          "(the 21.8 default is only roughly V-band, not a "
                          "specific LSST filter -- turn this 'on' yourself "
                          "for fixed mode only if you also pick a "
                          "band-appropriate --natural-sky-mag-arcsec2).")
    ap.add_argument("--out", default=None,
                     help="CSV path for the timestep-by-timestep output "
                          "(default: night_sky_brightness_{date}.csv)")
    ap.add_argument("--use-mock-backend", action="store_true",
                     help="Use the unphysical MockBrightnessBackend instead "
                          "of the real data_center_20deg -- for testing this "
                          "pipeline without lumos/starlink/analysis installed.")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main():
    args, _ = build_arg_parser().parse_known_args()

    if args.use_mock_backend:
        print("[WARN] Using MockBrightnessBackend -- NOT physically "
              "calibrated, for pipeline testing only.")
        backend = MockBrightnessBackend()
    else:
        backend = RealBrightnessBackend()

    sky_baseline = None
    opsim_db_path = None
    if args.sky_baseline == "real":
        sky_baseline = RealSkyBaseline(
            band=args.band, nside=args.healpix_nside,
            data_path=args.rubin_sim_data_path,
        )
    elif args.sky_baseline == "opsim":
        opsim_db_path = args.opsim_db or _default_opsim_db_path()
        if not os.path.isfile(opsim_db_path):
            raise FileNotFoundError(
                f"--sky-baseline opsim: could not find {opsim_db_path!r}. "
                f"Pass --opsim-db explicitly, or place the .db file next "
                f"to this script."
            )

    if args.band_correction == "on":
        apply_band_correction = True
    elif args.band_correction == "off":
        apply_band_correction = False
    else:  # auto
        apply_band_correction = args.sky_baseline in ("real", "opsim")

    print(f"[INFO] Band correction (532nm -> {args.band}): "
          f"{'ON' if apply_band_correction else 'OFF'} "
          f"(--band-correction={args.band_correction}, "
          f"--sky-baseline={args.sky_baseline})")
    if apply_band_correction:
        print(f"[INFO] solar_color[{args.band!r}] = "
              f"{SOLAR_COLOR[base_band(args.band)]:+.3f}")

    df = run_night(
        date_str=args.date, tle_dir=args.tle_dir,
        time_step_sec=args.time_step_sec,
        elevation_cutoff_deg=args.elevation_cutoff_deg,
        natural_sky_mag_arcsec2=args.natural_sky_mag_arcsec2,
        backend=backend,
        night_sun_el_max_deg=args.night_sun_el_max_deg,
        sky_baseline=sky_baseline,
        opsim_db_path=opsim_db_path,
        opsim_band=args.band,
        apply_band_correction=apply_band_correction,
        verbose=args.verbose,
    )

    out_path = args.out or f"night_sky_brightness_{args.date}.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print(f"NIGHT SKY BRIGHTNESS SUMMARY -- {args.date}")
    print("=" * 70)
    print(f"  Elevation cutoff       : {args.elevation_cutoff_deg}°")
    if args.sky_baseline == "real":
        print(f"  Natural sky baseline   : real per-timestep SkyModelPre "
              f"forecast (band={args.band})")
        print(f"  Natural sky mag range  : {df['mu_natural'].min():.3f} to "
              f"{df['mu_natural'].max():.3f} mag/arcsec² "
              f"(mean {df['mu_natural'].mean():.3f})")
    elif args.sky_baseline == "opsim":
        print(f"  Natural sky baseline   : real opsim skyBrightness curve "
              f"(band={args.band}, db={opsim_db_path})")
        print(f"  Natural sky mag range  : {df['mu_natural'].min():.3f} to "
              f"{df['mu_natural'].max():.3f} mag/arcsec² "
              f"(mean {df['mu_natural'].mean():.3f})")
    else:
        print(f"  Natural sky baseline   : {args.natural_sky_mag_arcsec2} mag/arcsec² (fixed)")
    print(f"  Peak visible satellites: {df['n_visible'].max():,}")
    print(f"  Mean visible satellites: {df['n_visible'].mean():,.0f}")
    print(f"  Peak delta_mag         : {df['delta_mag'].max():+.5f}")
    print(f"  Mean delta_mag         : {df['delta_mag'].mean():+.5f}")
    print(f"  Peak %% increase        : {df['pct_increase'].max():.3f}%")
    print(f"  Mean %% increase        : {df['pct_increase'].mean():.3f}%")
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()