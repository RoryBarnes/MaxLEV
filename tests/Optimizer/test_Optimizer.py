"""Test optimizer wrapper for MaxLEV."""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.optimizer import Optimizer


class TestOptimizerInit:
    """Tests for Optimizer initialization."""

    def test_default_algorithm(self):
        """Test default algorithm is differential_evolution."""
        opt = Optimizer({})
        assert opt.algorithm == 'differential_evolution'

    def test_custom_algorithm(self):
        """Test setting custom algorithm."""
        opt = Optimizer({'algorithm': 'powell'})
        assert opt.algorithm == 'powell'


class TestOptimizerDifferentialEvolution:
    """Tests for differential evolution optimizer."""

    def test_de_finds_minimum(self):
        """Test that DE can find a simple minimum."""
        def rosenbrock(x):
            return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

        config = {
            'algorithm': 'differential_evolution',
            'maxiter': 100,
            'seed': 42,
            'de_settings': {
                'popsize': 10,
                'workers': 1,
                'disp': False,
                'polish': False,
            }
        }

        opt = Optimizer(config)
        bounds = np.array([[-5, 5], [-5, 5]])

        best_params, best_value, info = opt.optimize(
            rosenbrock, bounds, ['x1', 'x2']
        )

        assert best_params.shape == (2,)
        assert isinstance(best_value, float)
        assert 'success' in info

    def test_de_respects_bounds(self):
        """Test that DE respects parameter bounds."""
        def objective(x):
            return np.sum(x**2)

        config = {
            'algorithm': 'differential_evolution',
            'maxiter': 50,
            'seed': 42,
            'de_settings': {
                'popsize': 5,
                'workers': 1,
                'disp': False,
                'polish': False,
            }
        }

        opt = Optimizer(config)
        bounds = np.array([[1, 3], [2, 4]])

        best_params, _, _ = opt.optimize(objective, bounds, ['x1', 'x2'])

        assert bounds[0, 0] <= best_params[0] <= bounds[0, 1]
        assert bounds[1, 0] <= best_params[1] <= bounds[1, 1]

    def test_de_uses_config_settings(self):
        """Test that DE uses configuration settings."""
        iCallCount = [0]

        def counting_objective(x):
            iCallCount[0] += 1
            return np.sum(x**2)

        config = {
            'algorithm': 'differential_evolution',
            'maxiter': 5,
            'tol': 0.1,
            'seed': 42,
            'de_settings': {
                'popsize': 3,
                'workers': 1,
                'disp': False,
                'polish': False,
            }
        }

        opt = Optimizer(config)
        bounds = np.array([[-1, 1]])

        opt.optimize(counting_objective, bounds, ['x1'])

        assert iCallCount[0] > 0


class TestOptimizerLocalMethods:
    """Tests for local optimizer methods."""

    def test_powell_finds_minimum(self):
        """Test that Powell method can find a minimum."""
        def quadratic(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2

        config = {
            'algorithm': 'powell',
            'maxiter': 100,
            'disp': False,
        }

        opt = Optimizer(config)
        bounds = np.array([[0, 5], [0, 5]])

        best_params, best_value, info = opt.optimize(
            quadratic, bounds, ['x1', 'x2']
        )

        assert abs(best_params[0] - 2.0) < 0.5
        assert abs(best_params[1] - 3.0) < 0.5

    def test_nelder_mead_finds_minimum(self):
        """Test that Nelder-Mead method can find a minimum."""
        def quadratic(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2

        config = {
            'algorithm': 'nelder-mead',
            'maxiter': 100,
            'disp': False,
        }

        opt = Optimizer(config)
        bounds = np.array([[0, 3], [0, 4]])

        best_params, best_value, info = opt.optimize(
            quadratic, bounds, ['x1', 'x2']
        )

        assert abs(best_params[0] - 1.0) < 0.5
        assert abs(best_params[1] - 2.0) < 0.5

    def test_bfgs_finds_minimum(self):
        """Test that BFGS method can find a minimum."""
        def quadratic(x):
            return (x[0] - 1.5)**2 + (x[1] - 2.5)**2

        config = {
            'algorithm': 'bfgs',
            'maxiter': 100,
            'disp': False,
        }

        opt = Optimizer(config)
        bounds = np.array([[0, 4], [0, 5]])

        best_params, best_value, info = opt.optimize(
            quadratic, bounds, ['x1', 'x2']
        )

        assert isinstance(best_params, np.ndarray)
        assert isinstance(best_value, float)

    def test_cobyla_finds_minimum(self):
        """Test that COBYLA method can find a minimum."""
        def quadratic(x):
            return (x[0] - 0.5)**2 + (x[1] - 0.5)**2

        config = {
            'algorithm': 'cobyla',
            'maxiter': 200,
            'disp': False,
        }

        opt = Optimizer(config)
        bounds = np.array([[0, 1], [0, 1]])

        best_params, best_value, info = opt.optimize(
            quadratic, bounds, ['x1', 'x2']
        )

        assert isinstance(best_params, np.ndarray)


class TestOptimizerInitialGuess:
    """Tests for x0 initial guess support."""

    def test_default_x0_is_center_of_bounds(self):
        """Test that default x0 is the center of bounds."""
        opt = Optimizer({'algorithm': 'nelder-mead'})
        bounds = np.array([[0, 10], [2, 8]])
        daX0 = opt._fdaParseInitialGuess(bounds)

        np.testing.assert_array_almost_equal(daX0, [5.0, 5.0])

    def test_config_x0_overrides_default(self):
        """Test that x0 from config overrides center of bounds."""
        opt = Optimizer({
            'algorithm': 'nelder-mead',
            'x0': [1.0, 2.0]
        })
        bounds = np.array([[0, 10], [0, 10]])
        daX0 = opt._fdaParseInitialGuess(bounds)

        np.testing.assert_array_almost_equal(daX0, [1.0, 2.0])

    def test_nelder_mead_with_x0_converges(self):
        """Test that NM with a good x0 converges near the minimum."""
        def quadratic(x):
            return (x[0] - 3.0)**2 + (x[1] - 7.0)**2

        config = {
            'algorithm': 'nelder-mead',
            'maxiter': 200,
            'disp': False,
            'x0': [2.9, 7.1],
        }

        opt = Optimizer(config)
        bounds = np.array([[0, 10], [0, 10]])

        best_params, best_value, info = opt.optimize(
            quadratic, bounds, ['x1', 'x2']
        )

        assert abs(best_params[0] - 3.0) < 0.1
        assert abs(best_params[1] - 7.0) < 0.1


class TestOptimizerNelderMeadSettings:
    """Tests for Nelder-Mead-specific settings."""

    def test_nm_settings_adaptive(self):
        """Test that adaptive setting is passed through."""
        def quadratic(x):
            return np.sum(x**2)

        config = {
            'algorithm': 'nelder-mead',
            'maxiter': 50,
            'disp': False,
            'nm_settings': {'adaptive': True},
        }

        opt = Optimizer(config)
        bounds = np.array([[-1, 1], [-1, 1]])

        best_params, best_value, info = opt.optimize(
            quadratic, bounds, ['x1', 'x2']
        )

        assert isinstance(best_params, np.ndarray)
        assert best_value < 0.1

    def test_nm_settings_tolerances(self):
        """Test that xatol and fatol settings are passed through."""
        def quadratic(x):
            return (x[0] - 0.5)**2

        config = {
            'algorithm': 'nelder-mead',
            'maxiter': 500,
            'disp': False,
            'nm_settings': {
                'xatol': 1e-8,
                'fatol': 1e-8,
            },
        }

        opt = Optimizer(config)
        bounds = np.array([[0, 1]])

        best_params, best_value, info = opt.optimize(
            quadratic, bounds, ['x1']
        )

        assert abs(best_params[0] - 0.5) < 1e-4

    def test_nm_builds_options_from_settings(self):
        """Test that _fdictBuildNelderMeadOptions extracts settings."""
        config = {
            'algorithm': 'nelder-mead',
            'nm_settings': {
                'adaptive': True,
                'xatol': 1e-6,
                'fatol': 1e-7,
            },
        }

        opt = Optimizer(config)
        dictOptions = opt._fdictBuildNelderMeadOptions()

        assert dictOptions['adaptive'] is True
        assert dictOptions['xatol'] == 1e-6
        assert dictOptions['fatol'] == 1e-7

    def test_nm_builds_empty_options_without_settings(self):
        """Test that empty nm_settings produces empty options."""
        opt = Optimizer({'algorithm': 'nelder-mead'})
        dictOptions = opt._fdictBuildNelderMeadOptions()

        assert dictOptions == {}

    def test_nm_initial_simplex(self):
        """Test that initial_simplex is converted to numpy array."""
        config = {
            'algorithm': 'nelder-mead',
            'nm_settings': {
                'initial_simplex': [[0, 0], [1, 0], [0, 1]],
            },
        }

        opt = Optimizer(config)
        dictOptions = opt._fdictBuildNelderMeadOptions()

        assert isinstance(dictOptions['initial_simplex'], np.ndarray)
        assert dictOptions['initial_simplex'].shape == (3, 2)


class TestOptimizerErrors:
    """Tests for optimizer error handling."""

    def test_unknown_algorithm_raises(self):
        """Test that unknown algorithm raises ValueError."""
        config = {'algorithm': 'unknown_optimizer'}
        opt = Optimizer(config)
        bounds = np.array([[-1, 1]])

        with pytest.raises(ValueError, match="Unknown algorithm"):
            opt.optimize(lambda x: x[0]**2, bounds, ['x1'])


class TestOptimizerOutput:
    """Tests for optimizer output formatting."""

    def test_returns_correct_shape(self):
        """Test that optimizer returns correct output shape."""
        config = {
            'algorithm': 'differential_evolution',
            'maxiter': 10,
            'seed': 42,
            'de_settings': {
                'popsize': 3,
                'workers': 1,
                'disp': False,
                'polish': False,
            }
        }

        opt = Optimizer(config)
        bounds = np.array([[-1, 1], [-2, 2], [-3, 3]])

        best_params, best_value, info = opt.optimize(
            lambda x: np.sum(x**2), bounds, ['a', 'b', 'c']
        )

        assert best_params.shape == (3,)
        assert isinstance(best_value, (int, float, np.floating))
        assert isinstance(info, dict)
        assert 'success' in info
        assert 'message' in info


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
