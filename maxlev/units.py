"""
Unit handling utilities for MaxLEV.

Parses unit strings and converts them to astropy units for use with vplanet_inference.
"""

import astropy.units as u
import vplanet  # noqa: F401 - registers VPlanet custom units (sec, EMAGMOM, etc.)
from typing import Dict


DICT_UNIT_MAP = {
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
    'kelvin': u.K,
    'celsius': u.deg_C,
    'dimensionless': u.dimensionless_unscaled,
}


def _fParseDexUnit(sUnitString: str) -> u.Unit:
    """Parse a dex(...) unit string into an astropy LogUnit."""
    if not sUnitString.endswith(')'):
        raise ValueError(f"Invalid dex unit format: {sUnitString}")
    sInner = sUnitString[4:-1]
    return u.dex(fParseUnitString(sInner))


def fParseUnitString(sUnitString: str) -> u.Unit:
    """
    Parse unit string into astropy unit.

    Supports common astrophysical units, dex(...) log units,
    empty strings (dimensionless), and astropy-parseable strings.

    Args:
        sUnitString: Unit string from configuration

    Returns:
        Astropy unit object

    Raises:
        ValueError: If unit string cannot be parsed
    """
    sUnitString = sUnitString.strip()

    if sUnitString == '':
        return u.dimensionless_unscaled

    if sUnitString.lower().startswith('dex('):
        return _fParseDexUnit(sUnitString)

    sKey = sUnitString.lower()
    if sKey in DICT_UNIT_MAP:
        return DICT_UNIT_MAP[sKey]

    try:
        return u.Unit(sUnitString)
    except ValueError:
        raise ValueError(f"Unknown unit: {sUnitString}")


def fdictBuildInparams(listParameters: list) -> Dict[str, u.Unit]:
    """
    Build the inparams dictionary for VplanetModel.

    Args:
        listParameters: List of ParameterConfig objects

    Returns:
        Dict mapping parameter names to astropy units
    """
    dictInparams = {}
    for param in listParameters:
        unitParsed = fParseUnitString(param.units)
        for sExpandedName in param.flistExpandedNames():
            dictInparams[sExpandedName] = unitParsed
    return dictInparams


def fdictBuildOutparams(listOutputs: list) -> Dict[str, u.Unit]:
    """
    Build the outparams dictionary for VplanetModel.

    Args:
        listOutputs: List of OutputConfig objects

    Returns:
        Dict mapping output names to astropy units
    """
    return {o.name: fParseUnitString(o.units) for o in listOutputs}
