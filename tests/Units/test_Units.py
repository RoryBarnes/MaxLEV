"""Test unit handling utilities for MaxLEV."""

import pytest
import sys
from pathlib import Path
import astropy.units as u

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.units import parse_unit_string, build_inparams_dict, build_outparams_dict
from maxlev.config import ParameterConfig, OutputConfig


class TestParseUnitString:
    """Tests for parse_unit_string function."""

    def test_parse_msun(self):
        """Test parsing solar mass unit."""
        result = parse_unit_string('Msun')
        assert result == u.Msun

    def test_parse_msun_lowercase(self):
        """Test parsing solar mass unit (lowercase)."""
        result = parse_unit_string('msun')
        assert result == u.Msun

    def test_parse_lsun(self):
        """Test parsing solar luminosity unit."""
        result = parse_unit_string('Lsun')
        assert result == u.Lsun

    def test_parse_rsun(self):
        """Test parsing solar radius unit."""
        result = parse_unit_string('Rsun')
        assert result == u.Rsun

    def test_parse_gyr(self):
        """Test parsing gigayear unit."""
        result = parse_unit_string('Gyr')
        assert result == u.Gyr

    def test_parse_myr(self):
        """Test parsing megayear unit."""
        result = parse_unit_string('Myr')
        assert result == u.Myr

    def test_parse_yr(self):
        """Test parsing year unit."""
        result = parse_unit_string('yr')
        assert result == u.yr

    def test_parse_day(self):
        """Test parsing day unit."""
        result = parse_unit_string('day')
        assert result == u.day

    def test_parse_d_as_day(self):
        """Test parsing 'd' as day unit."""
        result = parse_unit_string('d')
        assert result == u.day

    def test_parse_s(self):
        """Test parsing second unit."""
        result = parse_unit_string('s')
        assert result == u.s

    def test_parse_kg(self):
        """Test parsing kilogram unit."""
        result = parse_unit_string('kg')
        assert result == u.kg

    def test_parse_m(self):
        """Test parsing meter unit."""
        result = parse_unit_string('m')
        assert result == u.m

    def test_parse_au(self):
        """Test parsing AU unit."""
        result = parse_unit_string('au')
        assert result == u.au

    def test_parse_rearth(self):
        """Test parsing Earth radius unit."""
        result = parse_unit_string('Rearth')
        assert result == u.Rearth

    def test_parse_mearth(self):
        """Test parsing Earth mass unit."""
        result = parse_unit_string('Mearth')
        assert result == u.Mearth

    def test_parse_dimensionless(self):
        """Test parsing dimensionless unit."""
        result = parse_unit_string('dimensionless')
        assert result == u.dimensionless_unscaled

    def test_parse_dex_dimensionless(self):
        """Test parsing dex(dimensionless) unit."""
        result = parse_unit_string('dex(dimensionless)')
        assert result == u.dex(u.dimensionless_unscaled)

    def test_parse_dex_s(self):
        """Test parsing dex(s) unit."""
        result = parse_unit_string('dex(s)')
        assert result == u.dex(u.s)

    def test_parse_dex_yr(self):
        """Test parsing dex(yr) unit."""
        result = parse_unit_string('dex(yr)')
        assert result == u.dex(u.yr)

    def test_parse_with_whitespace(self):
        """Test parsing unit string with leading/trailing whitespace."""
        result = parse_unit_string('  Msun  ')
        assert result == u.Msun

    def test_parse_invalid_dex_format(self):
        """Test that invalid dex format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dex unit format"):
            parse_unit_string('dex(Msun')

    def test_parse_unknown_unit(self):
        """Test that unknown units raise ValueError."""
        with pytest.raises(ValueError, match="Unknown unit"):
            parse_unit_string('unknown_unit_xyz')

    def test_parse_astropy_unit_directly(self):
        """Test parsing units that astropy can handle directly."""
        result = parse_unit_string('km')
        assert result == u.km

    def test_parse_compound_unit(self):
        """Test parsing compound astropy units."""
        result = parse_unit_string('km/s')
        assert result == u.km / u.s


class TestBuildInparamsDict:
    """Tests for build_inparams_dict function."""

    def test_builds_dict_from_parameters(self):
        """Test building inparams dictionary."""
        parameters = [
            ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun'),
            ParameterConfig(name='star.dAge', bounds=(1.0, 10.0), units='Gyr'),
        ]

        result = build_inparams_dict(parameters)

        assert 'star.dMass' in result
        assert 'star.dAge' in result
        assert result['star.dMass'] == u.Msun
        assert result['star.dAge'] == u.Gyr

    def test_empty_parameters(self):
        """Test building dict from empty parameter list."""
        result = build_inparams_dict([])
        assert result == {}

    def test_handles_dex_units(self):
        """Test building dict with dex units."""
        parameters = [
            ParameterConfig(
                name='star.dSatXUVFrac',
                bounds=(-5, -2),
                units='dex(dimensionless)'
            )
        ]

        result = build_inparams_dict(parameters)

        assert result['star.dSatXUVFrac'] == u.dex(u.dimensionless_unscaled)


class TestBuildOutparamsDict:
    """Tests for build_outparams_dict function."""

    def test_builds_dict_from_outputs(self):
        """Test building outparams dictionary."""
        outputs = [
            OutputConfig(name='final.star.Luminosity', units='Lsun'),
            OutputConfig(name='final.planet.Mass', units='Mearth'),
        ]

        result = build_outparams_dict(outputs)

        assert 'final.star.Luminosity' in result
        assert 'final.planet.Mass' in result
        assert result['final.star.Luminosity'] == u.Lsun
        assert result['final.planet.Mass'] == u.Mearth

    def test_empty_outputs(self):
        """Test building dict from empty output list."""
        result = build_outparams_dict([])
        assert result == {}

    def test_handles_conversion_factor(self):
        """Test that conversion_factor is handled (stored in OutputConfig)."""
        outputs = [
            OutputConfig(
                name='final.earth.HeatFlow',
                units='W',
                conversion_factor=1e12
            )
        ]

        result = build_outparams_dict(outputs)

        assert result['final.earth.HeatFlow'] == u.W


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
