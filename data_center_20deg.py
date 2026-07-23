"""
data_center.py
──────────────────────────────────────────────────────────────────────────────
Satellite Brightness Calculator

Key changes from original:
    1. Solar panel normal points toward the Sun by default.
    2. Panel can be offset by PANEL_OFFSET_DEG degrees from Sun direction
       toward nadir using Rodrigues rotation formula.

    This simulates brightness mitigation strategies where satellites
    deliberately tilt their solar panels away from perfectly facing the Sun.

    Rotation axis: k = Sun_vector × nadir_vector
    → panel tilts in the plane containing Sun and nadir (toward Earth)
    → dot(panel_normal, sun_direction) = cos(PANEL_OFFSET_DEG)
    → e.g. 20° offset → panel receives cos(20°) = 94% of max solar flux
    → reflected light toward observer is reduced

    PANEL_OFFSET_DEG = 0   → panel faces Sun exactly (max brightness)
    PANEL_OFFSET_DEG = 20  → panel tilted 20° toward nadir (dimmer)
    PANEL_OFFSET_DEG = 90  → panel faces nadir (no direct Sun reflection)

All function signatures and return values are identical to the original.
Step4 and any other calling code needs no changes.
"""

import numpy as np
import lumos.conversions
import lumos.brdf.library
from lumos.brdf.library import BINOMIAL
from lumos.geometry import Surface
from starlink import satellitemodels
from analysis import calculator

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

BASE_PANEL_AREA    = 104.96   # m² per panel
BASE_POWER         = 25.0     # kW per panel
PANEL_OFFSET_DEG   = 20.0     # degrees to tilt panel from Sun toward nadir
                               # set to 0.0 to restore original behavior

DEFAULT_SATELLITE_MODEL = satellitemodels.get_surfaces()
DEFAULT_EARTH_BRDF      = lumos.brdf.library.PHONG(Kd=0.2, Ks=0.2, n=300)

RADIATOR_ALBEDO = 0.9
radiator_brdf = lumos.brdf.library.LAMBERTIAN(RADIATOR_ALBEDO)  # albedo = 0.9


def intensity_to_ab_mag(intensity, clip = True):
    """
    Converts from intensity to AB Magnitude.
    If clip is set to True, outputs below 12 AB Magnitude will be clipped.

    :param intensity: Intensity :math:`\\frac{W}{m^2}`
    :type intensity: :class:`np.ndarray` or float
    :param clip: Whether or not to clip very small intensities
    :type clip: bool, optional
    :return: AB Magnitude
    :rtype: :class:`np.ndarray` or float
    """
    SPEED_OF_LIGHT = 299792458.0  # m/s
    WAVELENGTH = 532e-9            # m (532 nm is the reference wavelength for AB magnitude)
    log_val = intensity * WAVELENGTH / (SPEED_OF_LIGHT * 3631e-26)
    if clip:
        log_val = np.clip(log_val, 10e-12, None)
    ab_mag = -2.5 * np.log10( log_val )
    return ab_mag



# ─────────────────────────────────────────────────────────────────────────────
# Solar array BRDF
# ─────────────────────────────────────────────────────────────────────────────

def get_solar_array_brdf():
    """Get the BRDF model for solar arrays."""
    B = np.array([[0.534, -20.409]])
    C = np.array([[-527.765, 1000., -676.579, 430.596, -175.806, 57.879]])
    return BINOMIAL(B, C, d=3.0, l1=-3)

# ─────────────────────────────────────────────────────────────────────────────
# Sun direction vector (no offset)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_sun_direction_vectors(sun_alt, sun_azi):
    """
    Convert Sun altitude and azimuth to 3D unit direction vector(s).

    Uses standard astronomical convention: azimuth from North toward East.

    Coordinate system (matches lumos/calculator):
        x = East   (cos(alt) * sin(azi))
        y = North  (cos(alt) * cos(azi))
        z = Up     (sin(alt))

    Parameters
    ----------
    sun_alt : float or array-like   Sun elevation in degrees
    sun_azi : float or array-like   Sun azimuth in degrees (N=0, E=90)

    Returns
    -------
    np.ndarray  shape (N, 3)  unit vectors pointing toward Sun
    """
    sun_alt = np.atleast_1d(np.asarray(sun_alt, dtype=float))
    sun_azi = np.atleast_1d(np.asarray(sun_azi, dtype=float))

    alt_rad = np.deg2rad(sun_alt)
    azi_rad = np.deg2rad(sun_azi)

    sun_x = np.cos(alt_rad) * np.sin(azi_rad)   # East
    sun_y = np.cos(alt_rad) * np.cos(azi_rad)   # North
    sun_z = np.sin(alt_rad)                      # Up

    directions = np.column_stack([sun_x, sun_y, sun_z])

    # Normalise
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    directions = directions / norms

    return directions   # shape (N, 3)

# ─────────────────────────────────────────────────────────────────────────────
# Panel normal with Rodrigues rotation offset toward nadir
# ─────────────────────────────────────────────────────────────────────────────

def calculate_panel_normal(sun_alt:      float,
                            sun_azi:      float,
                            offset_deg:   float = PANEL_OFFSET_DEG) -> np.ndarray:
    """
    Compute solar panel normal vector tilted offset_deg from Sun toward nadir.

    Uses Rodrigues rotation formula:
        v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1 - cos(θ))

    where:
        v = Sun unit vector (what panel would face with no offset)
        k = rotation axis = normalize(Sun × nadir)
            → axis perpendicular to both Sun and nadir directions
            → rotation tilts panel in the Sun-nadir plane
        θ = offset_deg in radians

    Physical meaning:
        offset_deg = 0  → panel faces Sun directly (max brightness)
        offset_deg = 20 → panel tilted 20° from Sun toward Earth
                          dot(panel, Sun) = cos(20°) = 0.940
        offset_deg = 90 → panel faces nadir (no direct Sun reflection)

    Parameters
    ----------
    sun_alt    : float   Sun elevation at observer (degrees)
    sun_azi    : float   Sun azimuth at observer (N=0 E=90, degrees)
    offset_deg : float   tilt angle from Sun toward nadir (degrees)

    Returns
    -------
    np.ndarray  shape (3,)  panel normal unit vector [x, y, z]
    """
    # ── Sun direction unit vector ──────────────────────────────────────────
    sun_dir = calculate_sun_direction_vectors(sun_alt, sun_azi)[0]  # (3,)

    # ── If no offset, return Sun direction directly ────────────────────────
    if abs(offset_deg) < 1e-6:
        return sun_dir

    # ── Nadir direction (toward Earth center) ─────────────────────────────
    # In observer frame: nadir = -zenith = -ẑ = [0, 0, -1]
    nadir = np.array([0.0, 0.0, -1.0])

    # ── Rotation axis: k = normalize(Sun × nadir) ─────────────────────────
    # This axis is perpendicular to both Sun and nadir
    # Rotation around k tilts panel in the Sun-nadir plane
    k = np.cross(sun_dir, nadir)
    k_norm = np.linalg.norm(k)

    if k_norm < 1e-8:
        # Sun is along nadir (directly overhead or directly below)
        # → Sun and nadir are parallel → cross product is zero
        # → No well-defined tilt direction → return Sun direction unchanged
        return sun_dir

    k = k / k_norm   # normalize rotation axis

    # ── Rodrigues rotation formula ────────────────────────────────────────
    # Rotates sun_dir by offset_deg around k
    theta = np.radians(offset_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    panel_normal = (
        sun_dir * cos_t
        + np.cross(k, sun_dir) * sin_t
        + k * np.dot(k, sun_dir) * (1.0 - cos_t)
    )

    # Normalize (should already be unit, but guard against float error)
    norm = np.linalg.norm(panel_normal)
    if norm > 1e-10:
        panel_normal = panel_normal / norm

    return panel_normal   # shape (3,)


def validate_panel_offset(sun_alt: float, sun_azi: float,
                           offset_deg: float = PANEL_OFFSET_DEG):
    """
    Validation helper — prints dot products to verify offset is correct.
    dot(panel, sun)   should be cos(offset_deg)
    dot(panel, nadir) should be sin(offset_deg) approximately
    """
    sun_dir = calculate_sun_direction_vectors(sun_alt, sun_azi)[0]
    nadir   = np.array([0.0, 0.0, -1.0])
    panel   = calculate_panel_normal(sun_alt, sun_azi, offset_deg)

    dot_sun   = np.dot(panel, sun_dir)
    dot_nadir = np.dot(panel, nadir)
    expected  = np.cos(np.radians(offset_deg))

    print(f"Sun alt={sun_alt}°  azi={sun_azi}°  offset={offset_deg}°")
    print(f"  Sun direction  : {sun_dir.round(4)}")
    print(f"  Panel normal   : {panel.round(4)}")
    print(f"  dot(panel,Sun) : {dot_sun:.6f}  (expected cos({offset_deg}°)={expected:.6f})")
    print(f"  dot(panel,nadir): {dot_nadir:.6f}")
    print(f"  Offset correct : {abs(dot_sun - expected) < 1e-5}")

# ─────────────────────────────────────────────────────────────────────────────
# Power → panel area
# ─────────────────────────────────────────────────────────────────────────────

def power_to_area(power_kw: float, continuous: bool = False) -> float:
    """
    Convert power requirement to solar panel area in m².
    """
    area = (power_kw / BASE_POWER) * BASE_PANEL_AREA
    if continuous:
        area *= 2
    return area

# ─────────────────────────────────────────────────────────────────────────────
# Build surfaces — solar panel normal = offset Sun direction
# ─────────────────────────────────────────────────────────────────────────────

def get_surfaces_with_solar_array(power_kw:    float,
                                   sun_altitude: float,
                                   sun_azimuth:  float,
                                   continuous:   bool  = False,
                                   offset_deg:   float = PANEL_OFFSET_DEG,
                                   ) -> list:
    """
    Build satellite surface list with solar array tilted offset_deg from Sun
    toward nadir using Rodrigues rotation.

    Parameters
    ----------
    power_kw     : float   power requirement in kW
    sun_altitude : float   Sun elevation at observer (degrees)
    sun_azimuth  : float   Sun azimuth at observer (N=0 E=90, degrees)
    continuous   : bool    if True doubles panel area
    offset_deg   : float   panel tilt from Sun toward nadir (degrees)

    Returns
    -------
    list of Surface objects
    """
    surfaces         = DEFAULT_SATELLITE_MODEL.copy()
    area             = power_to_area(power_kw, continuous)
    solar_array_brdf = get_solar_array_brdf()

    # Panel normal: Sun direction rotated offset_deg toward nadir
    panel_normal = calculate_panel_normal(sun_altitude, sun_azimuth, offset_deg)

    solar_array_surface = Surface(area, panel_normal, solar_array_brdf)
    surfaces.append(solar_array_surface)

    return surfaces

def surface_with_radiators(surfaces, wobble_deg=5.0, wobble_axis='y'):
    """
    Add radiator surfaces to the satellite model.

    Radiator normal is nominally along x (body frame), with a small
    wobble of up to +/- wobble_deg applied — representing mechanical
    flexing of the deployed radiator panel (per Tony Tyson's note:
    "+/- 5 deg or more in conops").

    The wobble tilts the x-axis normal within the y-z plane by rotating
    around either y or z (both lie in the y-z plane), so the radiator
    normal picks up a small y or z component instead of being purely x.

    Parameters
    ----------
    surfaces    : list of Surface objects
    wobble_deg  : float  tilt angle, degrees (use 0.0 for no wobble)
    wobble_axis : str    'y' or 'z' — which in-plane axis to rotate around

    Returns
    -------
    list of Surface objects with radiators added
    """
    radiator_area = 110.0  # m² per radiator face

    theta = np.radians(wobble_deg)

    if wobble_axis == 'y':
        # Rotate x toward z, around y-axis
        # x' = x*cos(theta) + z*sin(theta)
        radiator_normal_1 = np.array([np.cos(theta), 0.0,  np.sin(theta)])
        radiator_normal_2 = np.array([-np.cos(theta), 0.0, -np.sin(theta)])
    elif wobble_axis == 'z':
        # Rotate x toward y, around z-axis
        # x' = x*cos(theta) + y*sin(theta)
        radiator_normal_1 = np.array([np.cos(theta),  np.sin(theta), 0.0])
        radiator_normal_2 = np.array([-np.cos(theta), -np.sin(theta), 0.0])
    else:
        raise ValueError("wobble_axis must be 'y' or 'z'")

    # Normalize (should already be unit, but guard against float error)
    radiator_normal_1 /= np.linalg.norm(radiator_normal_1)
    radiator_normal_2 /= np.linalg.norm(radiator_normal_2)

    surfaces.append(Surface(radiator_area, radiator_normal_1, radiator_brdf))
    surfaces.append(Surface(radiator_area, radiator_normal_2, radiator_brdf))
    return surfaces

# ─────────────────────────────────────────────────────────────────────────────
# Main brightness calculation  (identical signature to original)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_brightness(sat_height,
                         sat_altitude,
                         sat_azimuth,
                         sun_altitude,
                         sun_azimuth,
                         power_kw,
                         continuous=False,
                         include_sun=True,
                         include_earthshine=False,
                         earth_panel_density=151,
                         earth_brdf=None,
                         offset_deg=PANEL_OFFSET_DEG):
    """
    Calculate satellite brightness for given geometry and power.

    Solar panel normal is tilted offset_deg from Sun direction toward nadir
    using Rodrigues rotation. This simulates brightness mitigation where
    satellites deliberately off-point their solar panels.

    Parameters
    ----------
    sat_height   : float              satellite height above ground in metres
    sat_altitude : float or array     satellite elevation angle(s) in degrees
    sat_azimuth  : float or array     satellite azimuth angle(s) in degrees
    sun_altitude : float              Sun elevation angle in degrees
    sun_azimuth  : float              Sun azimuth angle in degrees
    power_kw     : float              solar panel power in kW
    continuous   : bool               if True uses continuous power config
    include_sun          : bool       include direct sunlight (default True)
    include_earthshine   : bool       include earthshine (default False)
    earth_panel_density  : int        earth panel density (default 151)
    earth_brdf           : object     earth BRDF (default PHONG Kd=0.2)
    offset_deg   : float              panel tilt from Sun toward nadir
                                      (default = PANEL_OFFSET_DEG = 20°)

    Returns
    -------
    dict
        {
          'intensity'   : intensity values,
          'ab_magnitude': AB magnitude values,
          'area'        : solar panel area in m²,
          'power_type'  : 'instantaneous' or 'continuous',
          'sun_normal'  : panel normal unit vector [x, y, z],
          'offset_deg'  : panel offset angle used,
          'dot_panel_sun': dot(panel_normal, sun_direction) = cos(offset_deg),
        }
    """
    # ── Surfaces with offset panel ────────────────────────────────────────
    surfaces_with_panel = get_surfaces_with_solar_array(
        power_kw     = power_kw,
        sun_altitude = float(sun_altitude),
        sun_azimuth  = float(sun_azimuth),
        continuous   = continuous,
        offset_deg   = offset_deg,
    )
    surfaces = surface_with_radiators(surfaces_with_panel)

    if earth_brdf is None:
        earth_brdf = DEFAULT_EARTH_BRDF

    sat_altitude = np.atleast_1d(sat_altitude).astype(float)
    sat_azimuth  = np.atleast_1d(sat_azimuth).astype(float)

    # ── Intensity ─────────────────────────────────────────────────────────
    intensity = calculator.get_intensity_observer_frame(
        surfaces,
        np.ones(len(sat_altitude)) * sat_height,
        sat_altitude,
        sat_azimuth,
        sun_altitude,
        sun_azimuth,
        include_sun         = include_sun,
        include_earthshine  = include_earthshine,
        earth_panel_density = earth_panel_density,
        earth_brdf          = earth_brdf,
    )

    # ── AB magnitude ──────────────────────────────────────────────────────
    ab_magnitude = intensity_to_ab_mag(intensity)

    # ── Panel normal for validation ───────────────────────────────────────
    panel_normal = calculate_panel_normal(
        float(sun_altitude), float(sun_azimuth), offset_deg
    )
    sun_dir = calculate_sun_direction_vectors(
        float(sun_altitude), float(sun_azimuth)
    )[0]
    dot_panel_sun = float(np.dot(panel_normal, sun_dir))

    return {
        "intensity":     intensity,
        "ab_magnitude":  ab_magnitude,
        "area":          power_to_area(power_kw, continuous),
        "power_type":    "continuous" if continuous else "instantaneous",
        "sun_normal":    panel_normal.tolist(),
        "offset_deg":    offset_deg,
        "dot_panel_sun": dot_panel_sun,   # should be cos(offset_deg)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  data_center.py — Panel offset validation")
    print("=" * 60)

    # Validate offset geometry
    for sun_el in [-5, -10, -15, -20]:
        validate_panel_offset(sun_el, 180.0, PANEL_OFFSET_DEG)
        print()

    # Compare brightness: no offset vs 20° offset
    print("Brightness comparison (no offset vs 20° offset):")
    print(f"{'sun_el':>8}  {'mag_0deg':>10}  {'mag_20deg':>10}  {'delta_mag':>10}")
    print("-" * 45)

    for sun_el in [0, -5, -10, -15, -18, -20, -25, -30]:
        r0 = calculate_brightness(
            sat_height=700e3, sat_altitude=45.0, sat_azimuth=180.0,
            sun_altitude=sun_el, sun_azimuth=180.0,
            power_kw=400.0, continuous=False, offset_deg=0.0,
        )
        r20 = calculate_brightness(
            sat_height=700e3, sat_altitude=45.0, sat_azimuth=180.0,
            sun_altitude=sun_el, sun_azimuth=180.0,
            power_kw=400.0, continuous=False, offset_deg=20.0,
        )
        m0  = r0["ab_magnitude"][0]
        m20 = r20["ab_magnitude"][0]
        dm  = m20 - m0   # positive = fainter with offset
        print(f"{sun_el:>8}°  {m0:>10.4f}  {m20:>10.4f}  {dm:>+10.4f}")