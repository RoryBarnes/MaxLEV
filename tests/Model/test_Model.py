"""Test VPlanet model wrapper for MaxLEV."""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.model import MaxLEVModel, AllSimulationsFailedError
from maxlev.config import ParameterConfig, OutputConfig, ObservableConfig


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self):
        self.parameters = [
            ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun'),
            ParameterConfig(name='planet.dEcc', bounds=(0.0, 0.5), units='dimensionless'),
        ]
        self.outputs = [
            OutputConfig(name='final.planet.Ecc', units='dimensionless', conversion_factor=1.0),
            OutputConfig(name='final.star.Mass', units='Msun', conversion_factor=1.0),
        ]
        self.observables = [
            ObservableConfig(
                name='Eccentricity',
                type='direct',
                output='final.planet.Ecc',
                observed_value=0.1,
                uncertainty=0.02
            )
        ]
        self.vplanet = {
            'inpath': '/tmp/test',
            'executable': 'vplanet',
            'timeout': 10,
        }
        self.likelihood = {
            'failure_penalty': 1e10,
        }


class TestCheckBounds:
    """Tests for check_bounds method."""

    @patch('maxlev.model.vpi')
    def test_within_bounds_returns_true(self, mock_vpi):
        """Test that parameters within bounds return True."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([1.0, 0.2])
        assert model.check_bounds(theta) is True

    @patch('maxlev.model.vpi')
    def test_below_lower_bound_returns_false(self, mock_vpi):
        """Test that parameters below lower bounds return False."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([0.3, 0.2])  # star.dMass below 0.5
        assert model.check_bounds(theta) is False

    @patch('maxlev.model.vpi')
    def test_above_upper_bound_returns_false(self, mock_vpi):
        """Test that parameters above upper bounds return False."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([1.0, 0.6])  # planet.dEcc above 0.5
        assert model.check_bounds(theta) is False

    @patch('maxlev.model.vpi')
    def test_at_boundary_returns_true(self, mock_vpi):
        """Test that parameters exactly at bounds return True."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([0.5, 0.5])  # at lower and upper bounds
        assert model.check_bounds(theta) is True


class TestNegLogLikelihood:
    """Tests for neg_log_likelihood method."""

    @patch('maxlev.model.vpi')
    def test_returns_penalty_when_out_of_bounds(self, mock_vpi):
        """Test that out-of-bounds returns failure penalty."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([0.1, 0.2])  # out of bounds
        result = model.neg_log_likelihood(theta)

        assert result == 1e10

    @patch('maxlev.model.vpi')
    def test_returns_penalty_when_simulation_fails(self, mock_vpi):
        """Test that failed simulation returns failure penalty."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.side_effect = Exception("Simulation failed")
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([1.0, 0.2])
        result = model.neg_log_likelihood(theta)

        assert result == 1e10

    @patch('maxlev.model.vpi')
    def test_returns_penalty_for_nan_outputs(self, mock_vpi):
        """Test that NaN outputs return failure penalty."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.return_value = np.array([np.nan, 1.0])
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([1.0, 0.2])
        result = model.neg_log_likelihood(theta)

        assert result == 1e10

    @patch('maxlev.model.vpi')
    def test_returns_penalty_for_zero_outputs(self, mock_vpi):
        """Test that zero outputs return failure penalty."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.return_value = np.array([0.0, 1.0])
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([1.0, 0.2])
        result = model.neg_log_likelihood(theta)

        assert result == 1e10

    @patch('maxlev.model.vpi')
    def test_returns_penalty_for_negative_outputs(self, mock_vpi):
        """Test that negative outputs return failure penalty."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.return_value = np.array([-1.0, 1.0])
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([1.0, 0.2])
        result = model.neg_log_likelihood(theta)

        assert result == 1e10

    @patch('maxlev.model.vpi')
    def test_returns_likelihood_for_valid_run(self, mock_vpi):
        """Test that valid run returns computed likelihood."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.return_value = np.array([0.1, 1.0])
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        likelihood = MagicMock()
        likelihood.compute.return_value = 0.5

        observable_computer = MagicMock()
        observable_computer.compute.return_value = {'Eccentricity': 0.1}

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([1.0, 0.2])
        result = model.neg_log_likelihood(theta)

        assert result == 0.5
        likelihood.compute.assert_called_once()

    @patch('maxlev.model.vpi')
    def test_returns_penalty_when_observable_fails(self, mock_vpi):
        """Test that observable computation failure returns penalty."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.return_value = np.array([0.1, 1.0])
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        likelihood = MagicMock()

        observable_computer = MagicMock()
        observable_computer.compute.side_effect = Exception("Computation failed")

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([1.0, 0.2])
        result = model.neg_log_likelihood(theta)

        assert result == 1e10


class TestModelInit:
    """Tests for MaxLEVModel initialization."""

    @patch('maxlev.model.vpi')
    def test_sets_bounds_from_config(self, mock_vpi):
        """Test that bounds are set from config."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        expected = np.array([[0.5, 1.5], [0.0, 0.5]])
        np.testing.assert_array_equal(model.bounds, expected)

    @patch('maxlev.model.vpi')
    def test_sets_param_names_from_config(self, mock_vpi):
        """Test that parameter names are set from config."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        assert model.param_names == ['star.dMass', 'planet.dEcc']

    @patch('maxlev.model.vpi')
    def test_sets_timeout_from_config(self, mock_vpi):
        """Test that timeout is set from config."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        config.vplanet['timeout'] = 60

        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        assert model.iTimeout == 60

    @patch('maxlev.model.vpi')
    def test_sets_failure_penalty_from_config(self, mock_vpi):
        """Test that failure penalty is set from config."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        config.likelihood['failure_penalty'] = 1e5

        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        assert model.failure_penalty == 1e5


class TestConversionFactors:
    """Tests for conversion factor handling."""

    @patch('maxlev.model.vpi')
    def test_applies_conversion_factors(self, mock_vpi):
        """Test that conversion factors are applied to outputs."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.return_value = np.array([1.0, 2.0])  # alphabetically sorted
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        config.outputs = [
            OutputConfig(name='final.planet.Ecc', units='dimensionless', conversion_factor=10.0),
            OutputConfig(name='final.star.Mass', units='Msun', conversion_factor=100.0),
        ]

        likelihood = MagicMock()
        observable_computer = MagicMock()

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([1.0, 0.2])
        outputs = model.run_simulation(theta)

        assert outputs is not None
        assert outputs[0] == 10.0  # 1.0 * 10.0 (final.planet.Ecc comes first alphabetically)
        assert outputs[1] == 200.0  # 2.0 * 100.0


class TestFailureTracking:
    """Tests for simulation failure tracking and early abort."""

    @patch('maxlev.model.vpi')
    def test_tracks_simulation_failures(self, mock_vpi):
        """Test that simulation failures are counted."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.side_effect = Exception("VPLanet crashed")
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        config.likelihood['failure_check_window'] = 20
        model = MaxLEVModel(config, MagicMock(), MagicMock())

        theta = np.array([1.0, 0.2])
        model.neg_log_likelihood(theta)

        assert model.iSimulationCount == 1
        assert model.iSimulationFailureCount == 1

    @patch('maxlev.model.vpi')
    def test_aborts_after_all_simulations_fail(self, mock_vpi):
        """Test that AllSimulationsFailedError is raised."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.side_effect = Exception("VPLanet crashed")
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        config.likelihood['failure_check_window'] = 3
        model = MaxLEVModel(config, MagicMock(), MagicMock())

        theta = np.array([1.0, 0.2])
        model.neg_log_likelihood(theta)
        model.neg_log_likelihood(theta)
        with pytest.raises(AllSimulationsFailedError):
            model.neg_log_likelihood(theta)

    @patch('maxlev.model.vpi')
    def test_no_abort_when_some_succeed(self, mock_vpi):
        """Test that mixed success/failure does not abort."""
        iCallCount = [0]

        def fnAlternatingResults(theta, remove=True):
            iCallCount[0] += 1
            if iCallCount[0] % 2 == 0:
                raise Exception("Failure")
            return np.array([0.1, 1.0])

        mock_vpm = MagicMock()
        mock_vpm.run_model.side_effect = fnAlternatingResults
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        config.likelihood['failure_check_window'] = 3
        likelihood = MagicMock()
        likelihood.compute.return_value = 5.0
        observable_computer = MagicMock()
        observable_computer.compute.return_value = {'Eccentricity': 0.1}

        model = MaxLEVModel(config, likelihood, observable_computer)

        theta = np.array([1.0, 0.2])
        for _ in range(10):
            model.neg_log_likelihood(theta)

        assert model.iSimulationFailureCount < model.iSimulationCount

    @patch('maxlev.model.vpi')
    def test_bounds_violations_not_counted(self, mock_vpi):
        """Test that out-of-bounds calls do not increment counters."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        model = MaxLEVModel(config, MagicMock(), MagicMock())

        theta_oob = np.array([0.1, 0.2])
        model.neg_log_likelihood(theta_oob)

        assert model.iSimulationCount == 0
        assert model.iSimulationFailureCount == 0

    @patch('maxlev.model.vpi')
    def test_default_failure_check_window(self, mock_vpi):
        """Test that default failure_check_window is 10."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        model = MaxLEVModel(config, MagicMock(), MagicMock())

        assert model.iFailureCheckWindow == 10

    @patch('maxlev.model.vpi')
    def test_custom_failure_check_window(self, mock_vpi):
        """Test that failure_check_window is configurable."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        config.likelihood['failure_check_window'] = 5
        model = MaxLEVModel(config, MagicMock(), MagicMock())

        assert model.iFailureCheckWindow == 5


class TestNegLogPosterior:
    """Tests for fdNegLogPosterior method."""

    @patch('maxlev.model.vpi')
    def test_returns_likelihood_when_no_priors(self, mock_vpi):
        """Without priors, posterior equals likelihood."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.return_value = np.array([0.1, 1.0])
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        likelihood = MagicMock()
        likelihood.compute.return_value = 0.5
        observable_computer = MagicMock()
        observable_computer.compute.return_value = {'Eccentricity': 0.1}

        model = MaxLEVModel(config, likelihood, observable_computer)
        result = model.fdNegLogPosterior(np.array([1.0, 0.2]))

        assert result == 0.5

    @patch('maxlev.model.vpi')
    def test_adds_prior_to_likelihood(self, mock_vpi):
        """With priors, posterior = likelihood + prior."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.return_value = np.array([0.1, 1.0])
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        likelihood = MagicMock()
        likelihood.compute.return_value = 0.5
        observable_computer = MagicMock()
        observable_computer.compute.return_value = {'Eccentricity': 0.1}

        prior_collection = MagicMock()
        prior_collection.fdNegLogPrior.return_value = 0.3

        model = MaxLEVModel(
            config, likelihood, observable_computer,
            prior_collection=prior_collection,
        )
        result = model.fdNegLogPosterior(np.array([1.0, 0.2]))

        assert result == pytest.approx(0.8)

    @patch('maxlev.model.vpi')
    def test_returns_penalty_when_simulation_fails(self, mock_vpi):
        """Failed simulation returns penalty without computing prior."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.side_effect = Exception("Failed")
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        prior_collection = MagicMock()

        model = MaxLEVModel(
            config, MagicMock(), MagicMock(),
            prior_collection=prior_collection,
        )
        result = model.fdNegLogPosterior(np.array([1.0, 0.2]))

        assert result == 1e10
        prior_collection.fdNegLogPrior.assert_not_called()


class TestExpansionMap:
    """Tests for shared parameter expansion map and theta expansion."""

    def test_no_expansion_without_shared(self):
        """Test that expansion map is None when no shared params."""
        from maxlev.model import _fiaBuildExpansionMap
        params = [
            ParameterConfig(name='star.dMass', bounds=(0.5, 1.5), units='Msun'),
            ParameterConfig(name='planet.dEcc', bounds=(0.0, 0.5), units='dimensionless'),
        ]
        assert _fiaBuildExpansionMap(params) is None

    def test_expansion_map_shared(self):
        """Test expansion map for shared parameters."""
        from maxlev.model import _fiaBuildExpansionMap
        params = [
            ParameterConfig(
                name='dIceAlbedo', bounds=(0.4, 0.8), units='',
                bodies=['p1', 'p2', 'p3'],
            ),
            ParameterConfig(
                name='star.dMass', bounds=(0.5, 1.5), units='Msun',
            ),
        ]
        iaMap = _fiaBuildExpansionMap(params)
        assert iaMap == [0, 0, 0, 1]

    def test_expansion_map_all_shared(self):
        """Test expansion map with all shared parameters."""
        from maxlev.model import _fiaBuildExpansionMap
        params = [
            ParameterConfig(
                name='dIceAlbedo', bounds=(0.4, 0.8), units='',
                bodies=['p1', 'p2'],
            ),
            ParameterConfig(
                name='dDiffusion', bounds=(0.4, 0.8), units='',
                bodies=['p1', 'p2'],
            ),
        ]
        iaMap = _fiaBuildExpansionMap(params)
        assert iaMap == [0, 0, 1, 1]

    @patch('maxlev.model.vpi')
    def test_expand_theta_replicates_values(self, mock_vpi):
        """Test that _fdaExpandTheta replicates free params correctly."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        config.parameters = [
            ParameterConfig(
                name='dIceAlbedo', bounds=(0.4, 0.8), units='dimensionless',
                bodies=['p1', 'p2', 'p3'],
            ),
            ParameterConfig(
                name='star.dMass', bounds=(0.5, 1.5), units='Msun',
            ),
        ]
        config.outputs = [
            OutputConfig(name='final.p1.TGlobal', units='Celsius', conversion_factor=1.0),
        ]

        model = MaxLEVModel(config, MagicMock(), MagicMock())

        daTheta = np.array([0.6, 1.0])
        daExpanded = model._fdaExpandTheta(daTheta)

        np.testing.assert_array_equal(daExpanded, [0.6, 0.6, 0.6, 1.0])

    @patch('maxlev.model.vpi')
    def test_expand_theta_passthrough_when_no_shared(self, mock_vpi):
        """Test that _fdaExpandTheta returns input when no shared params."""
        mock_vpi.VplanetModel.return_value = MagicMock()

        config = MockConfig()
        model = MaxLEVModel(config, MagicMock(), MagicMock())

        daTheta = np.array([1.0, 0.2])
        daResult = model._fdaExpandTheta(daTheta)

        np.testing.assert_array_equal(daResult, daTheta)

    @patch('maxlev.model.vpi')
    def test_run_simulation_expands_theta(self, mock_vpi):
        """Test that run_simulation passes expanded theta to vpm."""
        mock_vpm = MagicMock()
        mock_vpm.run_model.return_value = np.array([10.0])
        mock_vpi.VplanetModel.return_value = mock_vpm

        config = MockConfig()
        config.parameters = [
            ParameterConfig(
                name='dIceAlbedo', bounds=(0.4, 0.8), units='dimensionless',
                bodies=['p1', 'p2'],
            ),
        ]
        config.outputs = [
            OutputConfig(name='final.p1.TGlobal', units='Celsius', conversion_factor=1.0),
        ]

        model = MaxLEVModel(config, MagicMock(), MagicMock())

        model.run_simulation(np.array([0.6]))

        daCalledTheta = mock_vpm.run_model.call_args[0][0]
        np.testing.assert_array_equal(daCalledTheta, [0.6, 0.6])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
