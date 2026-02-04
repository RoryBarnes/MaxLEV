"""Test invalid VPlanet output parameter name detection."""

import pytest
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.config import MaxLEVConfig
from tests.test_utils import FIXTURES_PATH, VPLANET_EXECUTABLE


def test_invalid_output_parameter():
    """Test that invalid VPlanet output parameter names are caught."""

    config_dict = {
        "name": "InvalidOutputTest",
        "vplanet": {
            "inpath": FIXTURES_PATH,
            "executable": VPLANET_EXECUTABLE,
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
                "name": "final.star.BogusOutput",
                "units": "dimensionless"
            }
        ],
        "observables": [
            {
                "name": "test_obs",
                "type": "direct",
                "output": "final.star.BogusOutput",
                "observed_value": 1.0,
                "uncertainty": 0.1
            }
        ],
        "likelihood": {"type": "gaussian"},
        "optimizer": {"algorithm": "differential_evolution"}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f)
        config_path = f.name

    try:
        config = MaxLEVConfig.from_json(config_path)
        errors = config.validate()

        assert len(errors) > 0, "Expected validation errors for invalid output parameter"

        error_messages = [err[0] for err in errors]
        has_invalid_output_error = any('BogusOutput' in msg or 'Invalid VPlanet output' in msg
                                       for msg in error_messages)
        assert has_invalid_output_error, f"Expected error about invalid output. Got: {error_messages}"

        print("✓ Invalid output parameter correctly detected")
        print(f"✓ Error message: {error_messages[0]}")

    finally:
        Path(config_path).unlink()


def test_invalid_output_in_derived_observable():
    """Test that invalid outputs in derived observable expressions are caught."""

    config_dict = {
        "name": "InvalidDerivedOutputTest",
        "vplanet": {
            "inpath": FIXTURES_PATH,
            "executable": VPLANET_EXECUTABLE,
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
                "name": "test_ratio",
                "type": "derived",
                "expression": "final.star.Luminosity / final.star.InvalidParameter",
                "observed_value": 1.0,
                "uncertainty": 0.1
            }
        ],
        "likelihood": {"type": "gaussian"},
        "optimizer": {"algorithm": "differential_evolution"}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f)
        config_path = f.name

    try:
        config = MaxLEVConfig.from_json(config_path)
        errors = config.validate()

        assert len(errors) > 0, "Expected validation errors for invalid output in expression"

        error_messages = [err[0] for err in errors]
        has_invalid_param_error = any('InvalidParameter' in msg for msg in error_messages)
        assert has_invalid_param_error, f"Expected error about InvalidParameter. Got: {error_messages}"

        print("✓ Invalid output in derived expression correctly detected")

    finally:
        Path(config_path).unlink()


def test_valid_output_passes():
    """Test that valid output parameter names pass validation."""

    config_dict = {
        "name": "ValidOutputTest",
        "vplanet": {
            "inpath": FIXTURES_PATH,
            "executable": VPLANET_EXECUTABLE,
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
        "optimizer": {"algorithm": "differential_evolution"}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f)
        config_path = f.name

    try:
        config = MaxLEVConfig.from_json(config_path)
        errors = config.validate()

        # Filter for output-specific errors
        output_errors = [err for err in errors if 'Luminosity' in err[0] and 'Invalid' in err[0]]

        assert len(output_errors) == 0, f"Valid output 'Luminosity' should pass. Got: {output_errors}"

        print("✓ Valid output 'Luminosity' passes validation")

    finally:
        Path(config_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
