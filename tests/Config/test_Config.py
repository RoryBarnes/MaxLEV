"""Test configuration loading and validation for MaxLEV."""

import pytest
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.config import (
    ParameterConfig,
    OutputConfig,
    ObservableConfig,
    MaxLEVConfig,
)


class TestParameterConfig:
    """Tests for ParameterConfig dataclass."""

    def test_file_name_extraction(self):
        """Test extracting file name from parameter name."""
        param = ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun')
        assert param.file_name == 'star'

    def test_param_name_extraction(self):
        """Test extracting param name from full name."""
        param = ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun')
        assert param.param_name == 'dMass'

    def test_param_name_no_dot(self):
        """Test param_name when no dot in name."""
        param = ParameterConfig(name='dMass', bounds=(0.5, 1.5), units='Msun')
        assert param.param_name == 'dMass'

    def test_is_log_space_with_dex(self):
        """Test is_log_space returns True for dex units."""
        param = ParameterConfig(name='star.dFrac', bounds=(-5, -2), units='dex(dimensionless)')
        assert param.is_log_space is True

    def test_is_log_space_without_dex(self):
        """Test is_log_space returns False for normal units."""
        param = ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun')
        assert param.is_log_space is False

    def test_is_log_space_case_insensitive(self):
        """Test is_log_space is case insensitive."""
        param = ParameterConfig(name='star.dFrac', bounds=(-5, -2), units='DEX(dimensionless)')
        assert param.is_log_space is True


class TestObservableConfig:
    """Tests for ObservableConfig dataclass."""

    def test_is_asymmetric_with_both_uncertainties(self):
        """Test is_asymmetric returns True when both uncertainties specified."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1,
            uncertainty_upper=0.2
        )
        assert obs.is_asymmetric is True

    def test_is_asymmetric_with_symmetric(self):
        """Test is_asymmetric returns False for symmetric uncertainties."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )
        assert obs.is_asymmetric is False

    def test_is_asymmetric_with_only_lower(self):
        """Test is_asymmetric returns False with only lower uncertainty."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1
        )
        assert obs.is_asymmetric is False

    def test_get_uncertainty_symmetric(self):
        """Test get_uncertainty returns symmetric uncertainty."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )
        assert obs.get_uncertainty(0.9) == 0.1
        assert obs.get_uncertainty(1.1) == 0.1

    def test_get_uncertainty_asymmetric_below(self):
        """Test get_uncertainty returns lower when computed < observed."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1,
            uncertainty_upper=0.2
        )
        assert obs.get_uncertainty(0.9) == 0.1

    def test_get_uncertainty_asymmetric_above(self):
        """Test get_uncertainty returns upper when computed >= observed."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1,
            uncertainty_upper=0.2
        )
        assert obs.get_uncertainty(1.1) == 0.2

    def test_get_uncertainty_asymmetric_equal(self):
        """Test get_uncertainty returns upper when computed == observed."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1,
            uncertainty_upper=0.2
        )
        assert obs.get_uncertainty(1.0) == 0.2


class TestMaxLEVConfigFromJson:
    """Tests for MaxLEVConfig.from_json method."""

    def test_loads_basic_config(self):
        """Test loading a basic JSON configuration."""
        config_data = {
            'name': 'TestRun',
            'vplanet': {'inpath': '/tmp/test'},
            'parameters': [
                {'name': 'star.dMass', 'bounds': [0.5, 1.5], 'units': 'Msun'}
            ],
            'outputs': [
                {'name': 'final.star.Mass', 'units': 'Msun'}
            ],
            'observables': [
                {
                    'name': 'Mass',
                    'type': 'direct',
                    'output': 'final.star.Mass',
                    'observed_value': 1.0,
                    'uncertainty': 0.1
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = MaxLEVConfig.from_json(config_path)
            assert config.name == 'TestRun'
            assert len(config.parameters) == 1
            assert len(config.outputs) == 1
            assert len(config.observables) == 1
        finally:
            Path(config_path).unlink()

    def test_loads_conversion_factor(self):
        """Test loading output with conversion_factor."""
        config_data = {
            'parameters': [],
            'outputs': [
                {'name': 'final.earth.HeatFlow', 'units': 'W', 'conversion_factor': 1e12}
            ],
            'observables': []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = MaxLEVConfig.from_json(config_path)
            assert config.outputs[0].conversion_factor == 1e12
        finally:
            Path(config_path).unlink()

    def test_loads_asymmetric_uncertainties(self):
        """Test loading observables with asymmetric uncertainties."""
        config_data = {
            'parameters': [],
            'outputs': [],
            'observables': [
                {
                    'name': 'Mass',
                    'type': 'direct',
                    'observed_value': 1.0,
                    'uncertainty_lower': 0.1,
                    'uncertainty_upper': 0.2
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = MaxLEVConfig.from_json(config_path)
            assert config.observables[0].is_asymmetric is True
        finally:
            Path(config_path).unlink()

    def test_default_values(self):
        """Test that defaults are applied for missing fields."""
        config_data = {
            'parameters': [],
            'outputs': [],
            'observables': []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = MaxLEVConfig.from_json(config_path)
            assert config.name == 'maxlev_run'
            assert config.likelihood == {'type': 'gaussian'}
            assert config.optimizer == {'algorithm': 'differential_evolution'}
        finally:
            Path(config_path).unlink()


class TestParameterConfigPrior:
    """Tests for ParameterConfig prior field."""

    def test_default_prior_is_none(self):
        """Prior defaults to None when not specified."""
        param = ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun')
        assert param.prior is None

    def test_stores_gaussian_prior(self):
        """Prior dict is stored when specified."""
        dictPrior = {"type": "gaussian", "mean": 1.0, "std": 0.1}
        param = ParameterConfig(
            name='star.dMass', bounds=(0.5, 1.5), units='Msun',
            prior=dictPrior,
        )
        assert param.prior == dictPrior

    def test_loads_prior_from_json(self):
        """Loads prior dict from JSON config."""
        config_data = {
            'parameters': [{
                'name': 'star.dMass',
                'bounds': [0.5, 1.5],
                'units': 'Msun',
                'prior': {
                    'type': 'asymmetric_gaussian',
                    'mean': 1.0,
                    'std_upper': 0.2,
                    'std_lower': 0.1,
                }
            }],
            'outputs': [],
            'observables': [],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = MaxLEVConfig.from_json(config_path)
            assert config.parameters[0].prior is not None
            assert config.parameters[0].prior['type'] == 'asymmetric_gaussian'
            assert config.parameters[0].prior['mean'] == 1.0
        finally:
            Path(config_path).unlink()

    def test_missing_prior_stays_none(self):
        """Parameters without 'prior' key have None prior."""
        config_data = {
            'parameters': [{
                'name': 'star.dMass',
                'bounds': [0.5, 1.5],
                'units': 'Msun',
            }],
            'outputs': [],
            'observables': [],
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = MaxLEVConfig.from_json(config_path)
            assert config.parameters[0].prior is None
        finally:
            Path(config_path).unlink()


class TestInpathResolution:
    """Tests for inpath resolution relative to config file."""

    def test_relative_inpath_resolves_to_config_directory(self):
        """Test that inpath '.' resolves to the config file's directory."""
        config_data = {
            'vplanet': {'inpath': '.'},
            'parameters': [],
            'outputs': [],
            'observables': []
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = MaxLEVConfig.from_json(config_path)
            sExpected = str(Path(config_path).resolve().parent)
            assert config.vplanet['inpath'] == sExpected
        finally:
            Path(config_path).unlink()

    def test_absolute_inpath_unchanged(self):
        """Test that an absolute inpath is not modified."""
        config_data = {
            'vplanet': {'inpath': '/tmp/test'},
            'parameters': [],
            'outputs': [],
            'observables': []
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = MaxLEVConfig.from_json(config_path)
            assert config.vplanet['inpath'] == str(
                Path('/tmp/test').resolve()
            )
        finally:
            Path(config_path).unlink()

    def test_subdirectory_inpath_resolves_correctly(self):
        """Test that a relative subdirectory inpath resolves correctly."""
        with tempfile.TemporaryDirectory() as sTmpDir:
            pathSubDir = Path(sTmpDir) / 'data'
            pathSubDir.mkdir()

            config_data = {
                'vplanet': {'inpath': 'data'},
                'parameters': [],
                'outputs': [],
                'observables': []
            }

            config_path = Path(sTmpDir) / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config_data, f)

            config = MaxLEVConfig.from_json(str(config_path))
            assert config.vplanet['inpath'] == str(pathSubDir.resolve())


def _fnMakeMockParam(sName='star.dMass'):
    """Create a mock parameter with correct shared-parameter attributes."""
    mock = MagicMock()
    mock.name = sName
    mock.bIsShared = False
    mock.flistExpandedNames.return_value = [sName]
    return mock


class TestMaxLEVConfigValidate:
    """Tests for MaxLEVConfig.validate method."""

    @patch('maxlev.config.validation.validate_all')
    def test_error_no_parameters(self, mock_validate):
        """Test validation error when no parameters defined."""
        mock_validate.return_value = []

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[],
            outputs=[MagicMock()],
            observables=[MagicMock(type='direct', output='test')],
            likelihood={},
            optimizer={},
            output_settings={}
        )

        errors = config.validate()
        error_messages = [e[0] for e in errors]
        assert any('No parameters' in msg for msg in error_messages)

    @patch('maxlev.config.validation.validate_all')
    def test_error_no_outputs(self, mock_validate):
        """Test validation error when no outputs defined."""
        mock_validate.return_value = []

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[_fnMakeMockParam()],
            outputs=[],
            observables=[MagicMock(type='direct', output='test')],
            likelihood={},
            optimizer={},
            output_settings={}
        )

        errors = config.validate()
        error_messages = [e[0] for e in errors]
        assert any('No outputs' in msg for msg in error_messages)

    @patch('maxlev.config.validation.validate_all')
    def test_error_no_observables(self, mock_validate):
        """Test validation error when no observables defined."""
        mock_validate.return_value = []

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[_fnMakeMockParam()],
            outputs=[MagicMock()],
            observables=[],
            likelihood={},
            optimizer={},
            output_settings={}
        )

        errors = config.validate()
        error_messages = [e[0] for e in errors]
        assert any('No observables' in msg for msg in error_messages)

    @patch('maxlev.config.validation.validate_all')
    def test_error_unknown_output_reference(self, mock_validate):
        """Test validation error for unknown output reference."""
        mock_validate.return_value = []

        output = MagicMock()
        output.name = 'final.star.Mass'

        obs = MagicMock()
        obs.type = 'direct'
        obs.output = 'nonexistent.output'
        obs.name = 'TestObs'
        obs.uncertainty = 0.1
        obs.is_asymmetric = False

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[_fnMakeMockParam()],
            outputs=[output],
            observables=[obs],
            likelihood={},
            optimizer={},
            output_settings={}
        )

        errors = config.validate()
        error_messages = [e[0] for e in errors]
        assert any('unknown output' in msg for msg in error_messages)

    @patch('maxlev.config.validation.validate_all')
    def test_error_invalid_observable_type(self, mock_validate):
        """Test validation error for invalid observable type."""
        mock_validate.return_value = []

        obs = MagicMock()
        obs.type = 'invalid_type'
        obs.output = 'test'
        obs.name = 'TestObs'
        obs.uncertainty = 0.1
        obs.is_asymmetric = False

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[_fnMakeMockParam()],
            outputs=[MagicMock()],
            observables=[obs],
            likelihood={},
            optimizer={},
            output_settings={}
        )

        errors = config.validate()
        error_messages = [e[0] for e in errors]
        assert any('invalid type' in msg for msg in error_messages)

    @patch('maxlev.config.validation.validate_all')
    def test_error_no_uncertainty(self, mock_validate):
        """Test validation error when observable has no uncertainty."""
        mock_validate.return_value = []

        output = MagicMock()
        output.name = 'test'

        obs = MagicMock()
        obs.type = 'direct'
        obs.output = 'test'
        obs.name = 'TestObs'
        obs.uncertainty = None
        obs.is_asymmetric = False

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[_fnMakeMockParam()],
            outputs=[output],
            observables=[obs],
            likelihood={},
            optimizer={},
            output_settings={}
        )

        errors = config.validate()
        error_messages = [e[0] for e in errors]
        assert any('no uncertainty' in msg for msg in error_messages)

    @patch('maxlev.config.validation.validate_all')
    def test_no_errors_valid_config(self, mock_validate):
        """Test that valid config has no errors."""
        mock_validate.return_value = []

        output = MagicMock()
        output.name = 'final.star.Mass'

        obs = MagicMock()
        obs.type = 'direct'
        obs.output = 'final.star.Mass'
        obs.name = 'Mass'
        obs.uncertainty = 0.1
        obs.is_asymmetric = False

        mockParam = MagicMock()
        mockParam.bIsShared = False
        mockParam.flistExpandedNames.return_value = ['star.dMass']

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[mockParam],
            outputs=[output],
            observables=[obs],
            likelihood={},
            optimizer={},
            output_settings={}
        )

        errors = config.validate()
        assert len(errors) == 0


class TestSharedParameters:
    """Tests for shared parameter (multi-body) configuration."""

    def test_bIsShared_true_with_bodies(self):
        """Test bIsShared is True when bodies list is provided."""
        param = ParameterConfig(
            name='dIceAlbedo', bounds=(0.4, 0.8), units='',
            bodies=['planet1', 'planet2', 'planet3'],
        )
        assert param.bIsShared is True

    def test_bIsShared_false_without_bodies(self):
        """Test bIsShared is False when bodies is None."""
        param = ParameterConfig(
            name='earth.dIceAlbedo', bounds=(0.4, 0.8), units='',
        )
        assert param.bIsShared is False

    def test_bIsShared_false_for_empty_bodies(self):
        """Test bIsShared is False when bodies list is empty."""
        param = ParameterConfig(
            name='dIceAlbedo', bounds=(0.4, 0.8), units='',
            bodies=[],
        )
        assert param.bIsShared is False

    def test_flistExpandedNames_shared(self):
        """Test expanded names for a shared parameter."""
        param = ParameterConfig(
            name='dIceAlbedo', bounds=(0.4, 0.8), units='',
            bodies=['planet1', 'planet2', 'planet3'],
        )
        expected = ['planet1.dIceAlbedo', 'planet2.dIceAlbedo',
                     'planet3.dIceAlbedo']
        assert param.flistExpandedNames() == expected

    def test_flistExpandedNames_non_shared(self):
        """Test expanded names for a non-shared parameter."""
        param = ParameterConfig(
            name='earth.dIceAlbedo', bounds=(0.4, 0.8), units='',
        )
        assert param.flistExpandedNames() == ['earth.dIceAlbedo']

    def test_param_name_shared(self):
        """Test param_name returns raw name for shared params."""
        param = ParameterConfig(
            name='dIceAlbedo', bounds=(0.4, 0.8), units='',
            bodies=['planet1', 'planet2'],
        )
        assert param.param_name == 'dIceAlbedo'

    def test_file_name_shared(self):
        """Test file_name returns first body for shared params."""
        param = ParameterConfig(
            name='dIceAlbedo', bounds=(0.4, 0.8), units='',
            bodies=['planet1', 'planet2'],
        )
        assert param.file_name == 'planet1'

    def test_loads_bodies_from_json(self):
        """Test loading shared parameter from JSON config."""
        config_data = {
            'parameters': [{
                'name': 'dIceAlbedo',
                'bodies': ['planet1', 'planet2'],
                'bounds': [0.4, 0.8],
                'units': '',
            }],
            'outputs': [],
            'observables': [],
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = MaxLEVConfig.from_json(config_path)
            assert config.parameters[0].bIsShared is True
            assert config.parameters[0].bodies == ['planet1', 'planet2']
        finally:
            Path(config_path).unlink()

    def test_missing_bodies_stays_none(self):
        """Test that parameters without 'bodies' key have None."""
        config_data = {
            'parameters': [{
                'name': 'earth.dIceAlbedo',
                'bounds': [0.4, 0.8],
                'units': '',
            }],
            'outputs': [],
            'observables': [],
        }

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = MaxLEVConfig.from_json(config_path)
            assert config.parameters[0].bodies is None
            assert config.parameters[0].bIsShared is False
        finally:
            Path(config_path).unlink()

    @patch('maxlev.config.validation.validate_all')
    def test_validate_rejects_dotted_name_with_bodies(self, mock_validate):
        """Test validation error for shared param with body.param name."""
        mock_validate.return_value = []

        output = MagicMock()
        output.name = 'final.planet1.TGlobal'

        obs = MagicMock()
        obs.type = 'direct'
        obs.output = 'final.planet1.TGlobal'
        obs.name = 'TGlobal'
        obs.uncertainty = 1.0
        obs.is_asymmetric = False

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[ParameterConfig(
                name='planet1.dIceAlbedo',
                bounds=(0.4, 0.8),
                units='',
                bodies=['planet1', 'planet2'],
            )],
            outputs=[output],
            observables=[obs],
            likelihood={},
            optimizer={},
            output_settings={},
        )

        errors = config.validate()
        error_messages = [e[0] for e in errors]
        assert any("must not use" in msg for msg in error_messages)

    @patch('maxlev.config.validation.validate_all')
    def test_validate_rejects_single_body(self, mock_validate):
        """Test validation error for shared param with fewer than 2 bodies."""
        mock_validate.return_value = []

        output = MagicMock()
        output.name = 'final.planet1.TGlobal'

        obs = MagicMock()
        obs.type = 'direct'
        obs.output = 'final.planet1.TGlobal'
        obs.name = 'TGlobal'
        obs.uncertainty = 1.0
        obs.is_asymmetric = False

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[ParameterConfig(
                name='dIceAlbedo',
                bounds=(0.4, 0.8),
                units='',
                bodies=['planet1'],
            )],
            outputs=[output],
            observables=[obs],
            likelihood={},
            optimizer={},
            output_settings={},
        )

        errors = config.validate()
        error_messages = [e[0] for e in errors]
        assert any("at least 2 bodies" in msg for msg in error_messages)

    @patch('maxlev.config.validation.validate_all')
    def test_validate_expands_shared_params(self, mock_validate):
        """Test that validation receives expanded parameter names."""
        mock_validate.return_value = []

        output = MagicMock()
        output.name = 'final.planet1.TGlobal'

        obs = MagicMock()
        obs.type = 'direct'
        obs.output = 'final.planet1.TGlobal'
        obs.name = 'TGlobal'
        obs.uncertainty = 1.0
        obs.is_asymmetric = False

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[ParameterConfig(
                name='dIceAlbedo',
                bounds=(0.4, 0.8),
                units='',
                bodies=['planet1', 'planet2'],
            )],
            outputs=[output],
            observables=[obs],
            likelihood={},
            optimizer={},
            output_settings={},
        )

        config.validate()

        call_args = mock_validate.call_args[0][0]
        param_names = [p['name'] for p in call_args['parameters']]
        assert 'planet1.dIceAlbedo' in param_names
        assert 'planet2.dIceAlbedo' in param_names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
