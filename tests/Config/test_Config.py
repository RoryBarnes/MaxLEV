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

    def test_sFileName_extraction(self):
        """Test extracting file name from parameter name."""
        param = ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun')
        assert param.sFileName == 'star'

    def test_sParamName_extraction(self):
        """Test extracting param name from full name."""
        param = ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun')
        assert param.sParamName == 'dMass'

    def test_sParamName_no_dot(self):
        """Test sParamName when no dot in name."""
        param = ParameterConfig(name='dMass', bounds=(0.5, 1.5), units='Msun')
        assert param.sParamName == 'dMass'

    def test_bIsLogSpace_with_dex(self):
        """Test bIsLogSpace returns True for dex units."""
        param = ParameterConfig(name='star.dFrac', bounds=(-5, -2), units='dex(dimensionless)')
        assert param.bIsLogSpace is True

    def test_bIsLogSpace_without_dex(self):
        """Test bIsLogSpace returns False for normal units."""
        param = ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun')
        assert param.bIsLogSpace is False

    def test_bIsLogSpace_case_insensitive(self):
        """Test bIsLogSpace is case insensitive."""
        param = ParameterConfig(name='star.dFrac', bounds=(-5, -2), units='DEX(dimensionless)')
        assert param.bIsLogSpace is True

    def test_fsSharedTag_when_shared(self):
        """Test fsSharedTag returns formatted tag for shared params."""
        param = ParameterConfig(
            name='dAlbedo', bounds=(0.2, 0.6), units='dimensionless',
            bodies=['planet1', 'planet2'],
        )
        assert param.fsSharedTag() == "  [shared: planet1, planet2]"

    def test_fsSharedTag_when_not_shared(self):
        """Test fsSharedTag returns empty string for non-shared params."""
        param = ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun')
        assert param.fsSharedTag() == ""


class TestObservableConfig:
    """Tests for ObservableConfig dataclass."""

    def test_bIsAsymmetric_with_both_uncertainties(self):
        """Test bIsAsymmetric returns True when both uncertainties specified."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1,
            uncertainty_upper=0.2
        )
        assert obs.bIsAsymmetric is True

    def test_bIsAsymmetric_with_symmetric(self):
        """Test bIsAsymmetric returns False for symmetric uncertainties."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )
        assert obs.bIsAsymmetric is False

    def test_bIsAsymmetric_with_only_lower(self):
        """Test bIsAsymmetric returns False with only lower uncertainty."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1
        )
        assert obs.bIsAsymmetric is False

    def test_fdGetUncertainty_symmetric(self):
        """Test fdGetUncertainty returns symmetric uncertainty."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty=0.1
        )
        assert obs.fdGetUncertainty(0.9) == 0.1
        assert obs.fdGetUncertainty(1.1) == 0.1

    def test_fdGetUncertainty_asymmetric_below(self):
        """Test fdGetUncertainty returns lower when computed < observed."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1,
            uncertainty_upper=0.2
        )
        assert obs.fdGetUncertainty(0.9) == 0.1

    def test_fdGetUncertainty_asymmetric_above(self):
        """Test fdGetUncertainty returns upper when computed >= observed."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1,
            uncertainty_upper=0.2
        )
        assert obs.fdGetUncertainty(1.1) == 0.2

    def test_fdGetUncertainty_asymmetric_equal(self):
        """Test fdGetUncertainty returns upper when computed == observed."""
        obs = ObservableConfig(
            name='test',
            type='direct',
            observed_value=1.0,
            uncertainty_lower=0.1,
            uncertainty_upper=0.2
        )
        assert obs.fdGetUncertainty(1.0) == 0.2


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
            assert config.observables[0].bIsAsymmetric is True
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
        obs.bIsAsymmetric = False

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
        obs.bIsAsymmetric = False

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
        obs.bIsAsymmetric = False

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
        obs.bIsAsymmetric = False

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
            name='dIceAlbedo', bounds=(0.4, 0.8), units='dimensionless',
            bodies=['planet1', 'planet2', 'planet3'],
        )
        assert param.bIsShared is True

    def test_bIsShared_false_without_bodies(self):
        """Test bIsShared is False when bodies is None."""
        param = ParameterConfig(
            name='earth.dIceAlbedo', bounds=(0.4, 0.8), units='dimensionless',
        )
        assert param.bIsShared is False

    def test_bIsShared_false_for_empty_bodies(self):
        """Test bIsShared is False when bodies list is empty."""
        param = ParameterConfig(
            name='dIceAlbedo', bounds=(0.4, 0.8), units='dimensionless',
            bodies=[],
        )
        assert param.bIsShared is False

    def test_flistExpandedNames_shared(self):
        """Test expanded names for a shared parameter."""
        param = ParameterConfig(
            name='dIceAlbedo', bounds=(0.4, 0.8), units='dimensionless',
            bodies=['planet1', 'planet2', 'planet3'],
        )
        expected = ['planet1.dIceAlbedo', 'planet2.dIceAlbedo',
                     'planet3.dIceAlbedo']
        assert param.flistExpandedNames() == expected

    def test_flistExpandedNames_non_shared(self):
        """Test expanded names for a non-shared parameter."""
        param = ParameterConfig(
            name='earth.dIceAlbedo', bounds=(0.4, 0.8), units='dimensionless',
        )
        assert param.flistExpandedNames() == ['earth.dIceAlbedo']

    def test_sParamName_shared(self):
        """Test sParamName returns raw name for shared params."""
        param = ParameterConfig(
            name='dIceAlbedo', bounds=(0.4, 0.8), units='dimensionless',
            bodies=['planet1', 'planet2'],
        )
        assert param.sParamName == 'dIceAlbedo'

    def test_sFileName_shared(self):
        """Test sFileName returns first body for shared params."""
        param = ParameterConfig(
            name='dIceAlbedo', bounds=(0.4, 0.8), units='dimensionless',
            bodies=['planet1', 'planet2'],
        )
        assert param.sFileName == 'planet1'

    def test_loads_bodies_from_json(self):
        """Test loading shared parameter from JSON config."""
        config_data = {
            'parameters': [{
                'name': 'dIceAlbedo',
                'bodies': ['planet1', 'planet2'],
                'bounds': [0.4, 0.8],
                'units': 'dimensionless',
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
                'units': 'dimensionless',
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
        obs.bIsAsymmetric = False

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[ParameterConfig(
                name='planet1.dIceAlbedo',
                bounds=(0.4, 0.8),
                units='dimensionless',
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
        obs.bIsAsymmetric = False

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[ParameterConfig(
                name='dIceAlbedo',
                bounds=(0.4, 0.8),
                units='dimensionless',
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
        obs.bIsAsymmetric = False

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[ParameterConfig(
                name='dIceAlbedo',
                bounds=(0.4, 0.8),
                units='dimensionless',
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

    @patch('maxlev.config.validation.validate_all')
    def test_validate_rejects_duplicate_bodies(self, mock_validate):
        """Test validation error for duplicate body names in shared param."""
        mock_validate.return_value = []

        output = MagicMock()
        output.name = 'final.planet1.TGlobal'

        obs = MagicMock()
        obs.type = 'direct'
        obs.output = 'final.planet1.TGlobal'
        obs.name = 'TGlobal'
        obs.uncertainty = 1.0
        obs.bIsAsymmetric = False

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[ParameterConfig(
                name='dIceAlbedo',
                bounds=(0.4, 0.8),
                units='dimensionless',
                bodies=['planet1', 'planet2', 'planet1'],
            )],
            outputs=[output],
            observables=[obs],
            likelihood={},
            optimizer={},
            output_settings={},
        )

        errors = config.validate()
        error_messages = [e[0] for e in errors]
        assert any("duplicate" in msg.lower() for msg in error_messages)

    @patch('maxlev.config.validation.validate_all')
    def test_validate_rejects_shared_nonshared_conflict(self, mock_validate):
        """Test validation error when shared and non-shared target same body.param."""
        mock_validate.return_value = []

        output = MagicMock()
        output.name = 'final.planet1.TGlobal'

        obs = MagicMock()
        obs.type = 'direct'
        obs.output = 'final.planet1.TGlobal'
        obs.name = 'TGlobal'
        obs.uncertainty = 1.0
        obs.bIsAsymmetric = False

        config = MaxLEVConfig(
            name='test',
            vplanet={},
            parameters=[
                ParameterConfig(
                    name='dIceAlbedo',
                    bounds=(0.4, 0.8),
                    units='dimensionless',
                    bodies=['planet1', 'planet2'],
                ),
                ParameterConfig(
                    name='planet1.dIceAlbedo',
                    bounds=(0.4, 0.8),
                    units='dimensionless',
                ),
            ],
            outputs=[output],
            observables=[obs],
            likelihood={},
            optimizer={},
            output_settings={},
        )

        errors = config.validate()
        error_messages = [e[0] for e in errors]
        assert any("conflict" in msg.lower() for msg in error_messages)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
