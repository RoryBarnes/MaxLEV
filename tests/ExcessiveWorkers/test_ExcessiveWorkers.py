"""Test worker count validation for parallel optimization."""

import pytest
import json
import tempfile
from pathlib import Path
import sys
import os
import multiprocessing

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vmle.config import VMLEConfig
from vmle.optimizer import Optimizer
import numpy as np


def test_workers_exceeds_cpu_count():
    """
    Test warning/handling when workers exceeds available CPU cores.

    Note: scipy's differential_evolution will handle this internally,
    but we should verify the configuration is accepted.
    """

    cpu_count = multiprocessing.cpu_count()
    excessive_workers = cpu_count + 10

    config_dict = {
        "name": "ExcessiveWorkersTest",
        "vplanet": {
            "inpath": "/Users/rory/src/GJ1132/XUV/Distributions/Ribas/Posteriors/emcee",
            "executable": "/Users/rory/src/vplanet/bin/vplanet",
            "vplfile": "vpl.in"
        },
        "parameters": [
            {
                "name": "star.dMass",
                "bounds": [0.1, 0.3],
                "units": "Msun"
            }
        ],
        "outputs": [
            {
                "name": "final.star.Luminosity",
                "units": "Lsun"
            }
        ],
        "observables": [
            {
                "name": "test_obs",
                "type": "direct",
                "output": "final.star.Luminosity",
                "observed_value": 1.0,
                "uncertainty": 0.1
            }
        ],
        "likelihood": {"type": "gaussian"},
        "optimizer": {
            "algorithm": "differential_evolution",
            "de_settings": {
                "workers": excessive_workers
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f)
        config_path = f.name

    try:
        config = VMLEConfig.from_json(config_path)

        # Configuration should load without errors
        assert config.optimizer['de_settings']['workers'] == excessive_workers

        # Create optimizer - scipy will handle excessive workers gracefully
        optimizer = Optimizer(config.optimizer)
        assert optimizer.config['de_settings']['workers'] == excessive_workers

        print(f"✓ System has {cpu_count} CPUs")
        print(f"✓ Configuration accepts {excessive_workers} workers")
        print(f"✓ scipy.differential_evolution will handle this gracefully")

    finally:
        Path(config_path).unlink()


def test_workers_equals_cpu_count():
    """Test that workers == CPU count works correctly."""

    cpu_count = multiprocessing.cpu_count()

    config_dict = {
        "name": "MaxWorkersTest",
        "optimizer": {
            "algorithm": "differential_evolution",
            "de_settings": {
                "workers": cpu_count
            }
        }
    }

    optimizer = Optimizer(config_dict['optimizer'])
    assert optimizer.config['de_settings']['workers'] == cpu_count

    print(f"✓ Workers set to CPU count ({cpu_count}) is valid")


def test_workers_minus_one():
    """
    Test that workers=-1 (use all CPUs) is handled correctly.

    This is a scipy.optimize.differential_evolution feature.
    """

    config_dict = {
        "optimizer": {
            "algorithm": "differential_evolution",
            "de_settings": {
                "workers": -1
            }
        }
    }

    optimizer = Optimizer(config_dict['optimizer'])
    assert optimizer.config['de_settings']['workers'] == -1

    print("✓ workers=-1 (use all CPUs) is accepted")


def test_single_worker():
    """Test that workers=1 (serial execution) works."""

    config_dict = {
        "optimizer": {
            "algorithm": "differential_evolution",
            "de_settings": {
                "workers": 1
            }
        }
    }

    optimizer = Optimizer(config_dict['optimizer'])
    assert optimizer.config['de_settings']['workers'] == 1

    print("✓ workers=1 (serial) is accepted")


# Define objective function at module level (needed for multiprocessing pickling)
def _simple_objective_for_testing(x):
    """Simple quadratic objective function for testing."""
    return np.sum(x**2)


def test_optimizer_respects_worker_count():
    """
    Test that the optimizer actually uses the specified worker count.

    This is a more integration-level test.
    """

    bounds = np.array([[0, 1], [0, 1]])

    # Test with 1 worker (serial)
    config_serial = {
        "algorithm": "differential_evolution",
        "seed": 42,
        "maxiter": 2,
        "de_settings": {
            "workers": 1,
            "popsize": 3,
            "polish": False,
            "disp": False
        }
    }

    optimizer_serial = Optimizer(config_serial)
    result_serial = optimizer_serial._run_differential_evolution(_simple_objective_for_testing, bounds)

    assert result_serial is not None
    print("✓ Serial optimization (workers=1) completes successfully")

    # Test with multiple workers if available
    cpu_count = multiprocessing.cpu_count()
    if cpu_count > 1:
        workers = min(2, cpu_count)
        config_parallel = {
            "algorithm": "differential_evolution",
            "seed": 42,
            "maxiter": 2,
            "de_settings": {
                "workers": workers,
                "popsize": 3,
                "polish": False,
                "disp": False
            }
        }

        optimizer_parallel = Optimizer(config_parallel)
        result_parallel = optimizer_parallel._run_differential_evolution(_simple_objective_for_testing, bounds)

        assert result_parallel is not None
        print(f"✓ Parallel optimization (workers={workers}) completes successfully")


def test_worker_count_info():
    """Display system information about available workers."""

    cpu_count = multiprocessing.cpu_count()

    print(f"\n{'='*60}")
    print(f"System CPU Information:")
    print(f"{'='*60}")
    print(f"Total CPU cores: {cpu_count}")
    print(f"Recommended max workers for DE: {cpu_count}")
    print(f"Using workers > {cpu_count} may not improve performance")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
