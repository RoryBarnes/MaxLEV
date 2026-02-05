"""
Unit handling utilities for MaxLEV.

Parses unit strings and converts them to astropy units for use with vplanet_inference.
"""

import astropy.units as u
import vplanet  # noqa: F401 - registers VPlanet custom units (sec, EMAGMOM, etc.)
from typing import Dict


def parse_unit_string(unit_str: str) -> u.Unit:
    """
    Parse unit string into astropy unit.

    Supports:
    - "Msun" → u.Msun
    - "Gyr" → u.Gyr
    - "Lsun" → u.Lsun
    - "dimensionless" → u.dimensionless_unscaled
    - "dex(dimensionless)" → u.dex(u.dimensionless_unscaled)
    - "dex(s)" → u.dex(u.s)

    Args:
        unit_str: Unit string from configuration

    Returns:
        Astropy unit object

    Raises:
        ValueError: If unit string cannot be parsed
    """
    unit_str = unit_str.strip()

    # Handle dex units (log10 space)
    if unit_str.lower().startswith('dex('):
        if not unit_str.endswith(')'):
            raise ValueError(f"Invalid dex unit format: {unit_str}")
        inner = unit_str[4:-1]  # Extract inner unit
        inner_unit = parse_unit_string(inner)
        return u.dex(inner_unit)

    # Map common unit names (case-insensitive)
    unit_map = {
        'msun': u.Msun,
        'lsun': u.Lsun,
        'rsun': u.Rsun,
        'gyr': u.Gyr,
        'myr': u.Myr,
        'yr': u.yr,
        'day': u.day,
        'd': u.day,
        's': u.s,
        'kg': u.kg,
        'm': u.m,
        'au': u.au,
        'rearth': u.Rearth,
        'mearth': u.Mearth,
        'dimensionless': u.dimensionless_unscaled,
    }

    key = unit_str.lower()
    if key in unit_map:
        return unit_map[key]

    # Try to parse as astropy unit directly
    try:
        return u.Unit(unit_str)
    except ValueError:
        raise ValueError(f"Unknown unit: {unit_str}")


def build_inparams_dict(parameters: list) -> Dict[str, u.Unit]:
    """
    Build the inparams dictionary for VplanetModel.

    Args:
        parameters: List of ParameterConfig objects

    Returns:
        Dict mapping parameter names to astropy units, e.g.:
        {
            "star.dMass": u.Msun,
            "star.dSatXUVFrac": u.dex(u.dimensionless_unscaled),
            ...
        }
    """
    return {p.name: parse_unit_string(p.units) for p in parameters}


def build_outparams_dict(outputs: list) -> Dict[str, u.Unit]:
    """
    Build the outparams dictionary for VplanetModel.

    Args:
        outputs: List of OutputConfig objects

    Returns:
        Dict mapping output names to astropy units, e.g.:
        {
            "final.star.Luminosity": u.Lsun,
            "final.star.LXUVStellar": u.Lsun,
        }
    """
    return {o.name: parse_unit_string(o.units) for o in outputs}
