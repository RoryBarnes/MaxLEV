"""Test unit handling utilities for MaxLEV."""

import pytest
import sys
from pathlib import Path
import astropy.units as u

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.units import fParseUnitString, fdictBuildInparams, fdictBuildOutparams
from maxlev.config import ParameterConfig, OutputConfig


class TestParseUnitString:
    """Tests for fParseUnitString function."""

    def test_parse_msun(self):
        """Test parsing solar mass unit."""
        result = fParseUnitString('Msun')
        assert result == u.Msun

    def test_parse_msun_lowercase(self):
        """Test parsing solar mass unit (lowercase)."""
        result = fParseUnitString('msun')
        assert result == u.Msun

    def test_parse_lsun(self):
        """Test parsing solar luminosity unit."""
        result = fParseUnitString('Lsun')
        assert result == u.Lsun

    def test_parse_rsun(self):
        """Test parsing solar radius unit."""
        result = fParseUnitString('Rsun')
        assert result == u.Rsun

    def test_parse_gyr(self):
        """Test parsing gigayear unit."""
        result = fParseUnitString('Gyr')
        assert result == u.Gyr

    def test_parse_myr(self):
        """Test parsing megayear unit."""
        result = fParseUnitString('Myr')
        assert result == u.Myr

    def test_parse_yr(self):
        """Test parsing year unit."""
        result = fParseUnitString('yr')
        assert result == u.yr

    def test_parse_day(self):
        """Test parsing day unit."""
        result = fParseUnitString('day')
        assert result == u.day

    def test_parse_d_as_day(self):
        """Test parsing 'd' as day unit."""
        result = fParseUnitString('d')
        assert result == u.day

    def test_parse_s(self):
        """Test parsing second unit."""
        result = fParseUnitString('s')
        assert result == u.s

    def test_parse_kg(self):
        """Test parsing kilogram unit."""
        result = fParseUnitString('kg')
        assert result == u.kg

    def test_parse_m(self):
        """Test parsing meter unit."""
        result = fParseUnitString('m')
        assert result == u.m

    def test_parse_au(self):
        """Test parsing AU unit."""
        result = fParseUnitString('au')
        assert result == u.au

    def test_parse_rearth(self):
        """Test parsing Earth radius unit."""
        result = fParseUnitString('Rearth')
        assert result == u.Rearth

    def test_parse_mearth(self):
        """Test parsing Earth mass unit."""
        result = fParseUnitString('Mearth')
        assert result == u.Mearth

    def test_parse_dimensionless(self):
        """Test parsing dimensionless unit."""
        result = fParseUnitString('dimensionless')
        assert result == u.dimensionless_unscaled

    def test_parse_dex_dimensionless(self):
        """Test parsing dex(dimensionless) unit."""
        result = fParseUnitString('dex(dimensionless)')
        assert result == u.dex(u.dimensionless_unscaled)

    def test_parse_dex_s(self):
        """Test parsing dex(s) unit."""
        result = fParseUnitString('dex(s)')
        assert result == u.dex(u.s)

    def test_parse_dex_yr(self):
        """Test parsing dex(yr) unit."""
        result = fParseUnitString('dex(yr)')
        assert result == u.dex(u.yr)

    def test_parse_with_whitespace(self):
        """Test parsing unit string with leading/trailing whitespace."""
        result = fParseUnitString('  Msun  ')
        assert result == u.Msun

    def test_parse_invalid_dex_format(self):
        """Test that invalid dex format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dex unit format"):
            fParseUnitString('dex(Msun')

    def test_parse_unknown_unit(self):
        """Test that unknown units raise ValueError."""
        with pytest.raises(ValueError, match="Unknown unit"):
            fParseUnitString('unknown_unit_xyz')

    def test_parse_astropy_unit_directly(self):
        """Test parsing units that astropy can handle directly."""
        result = fParseUnitString('km')
        assert result == u.km

    def test_parse_compound_unit(self):
        """Test parsing compound astropy units."""
        result = fParseUnitString('km/s')
        assert result == u.km / u.s

    def test_parse_empty_string(self):
        """Test that empty string returns dimensionless."""
        result = fParseUnitString('')
        assert result == u.dimensionless_unscaled

    def test_parse_whitespace_only(self):
        """Test that whitespace-only string returns dimensionless."""
        result = fParseUnitString('   ')
        assert result == u.dimensionless_unscaled


class TestBuildInparamsDict:
    """Tests for fdictBuildInparams function."""

    def test_builds_dict_from_parameters(self):
        """Test building inparams dictionary."""
        parameters = [
            ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun'),
            ParameterConfig(name='star.dAge', bounds=(1.0, 10.0), units='Gyr'),
        ]

        result = fdictBuildInparams(parameters)

        assert 'star.dMass' in result
        assert 'star.dAge' in result
        assert result['star.dMass'] == u.Msun
        assert result['star.dAge'] == u.Gyr

    def test_empty_parameters(self):
        """Test building dict from empty parameter list."""
        result = fdictBuildInparams([])
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

        result = fdictBuildInparams(parameters)

        assert result['star.dSatXUVFrac'] == u.dex(u.dimensionless_unscaled)


class TestBuildOutparamsDict:
    """Tests for fdictBuildOutparams function."""

    def test_builds_dict_from_outputs(self):
        """Test building outparams dictionary."""
        outputs = [
            OutputConfig(name='final.star.Luminosity', units='Lsun'),
            OutputConfig(name='final.planet.Mass', units='Mearth'),
        ]

        result = fdictBuildOutparams(outputs)

        assert 'final.star.Luminosity' in result
        assert 'final.planet.Mass' in result
        assert result['final.star.Luminosity'] == u.Lsun
        assert result['final.planet.Mass'] == u.Mearth

    def test_empty_outputs(self):
        """Test building dict from empty output list."""
        result = fdictBuildOutparams([])
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

        result = fdictBuildOutparams(outputs)

        assert result['final.earth.HeatFlow'] == u.W


class TestBuildInparamsDictShared:
    """Tests for fdictBuildInparams with shared parameters."""

    def test_expands_shared_parameter(self):
        """Test that shared param produces multiple dict entries."""
        parameters = [
            ParameterConfig(
                name='dIceAlbedo', bounds=(0.4, 0.8), units='dimensionless',
                bodies=['planet1', 'planet2', 'planet3'],
            ),
        ]

        result = fdictBuildInparams(parameters)

        assert len(result) == 3
        assert 'planet1.dIceAlbedo' in result
        assert 'planet2.dIceAlbedo' in result
        assert 'planet3.dIceAlbedo' in result
        for sKey in result:
            assert result[sKey] == u.dimensionless_unscaled

    def test_mixed_shared_and_non_shared(self):
        """Test dict with both shared and non-shared params."""
        parameters = [
            ParameterConfig(
                name='dIceAlbedo', bounds=(0.4, 0.8), units='dimensionless',
                bodies=['planet1', 'planet2'],
            ),
            ParameterConfig(
                name='star.dMass', bounds=(0.5, 1.5), units='Msun',
            ),
        ]

        result = fdictBuildInparams(parameters)

        assert len(result) == 3
        assert 'planet1.dIceAlbedo' in result
        assert 'planet2.dIceAlbedo' in result
        assert 'star.dMass' in result
        assert result['star.dMass'] == u.Msun

    def test_non_shared_unchanged(self):
        """Test that non-shared params still work as before."""
        parameters = [
            ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun'),
        ]

        result = fdictBuildInparams(parameters)

        assert len(result) == 1
        assert 'star.dMass' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
