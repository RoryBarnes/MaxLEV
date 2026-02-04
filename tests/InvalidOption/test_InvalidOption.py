"""Test invalid VPlanet option name detection."""

import pytest
import json
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.config import MaxLEVConfig
from tests.test_utils import FIXTURES_PATH, VPLANET_EXECUTABLE


def test_invalid_vplanet_option():
    """Test that invalid VPlanet parameter names are caught during validation."""

    # Create a config with an invalid parameter name
    config_dict = {
        "name": "InvalidOptionTest",
        "vplanet": {
            "inpath": FIXTURES_PATH,
            "executable": VPLANET_EXECUTABLE,
            "vplfile": "vpl.in"
        },
        "parameters": [
            {
                "name": "star.dInvalidOption",
                "bounds": [0.0, 1.0],
                "units": "dimensionless"
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

    # Write to temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_dict, f)
        config_path = f.name

    try:
        # Load configuration
        config = MaxLEVConfig.from_json(config_path)

        # Validate - should return errors
        errors = config.validate()

        # Should have at least one error
        assert len(errors) > 0, "Expected validation errors for invalid parameter name"

        # Check that error mentions the invalid parameter
        error_messages = [err[0] for err in errors]
        has_invalid_param_error = any('dInvalidOption' in msg or 'Invalid VPlanet parameter' in msg
                                      for msg in error_messages)
        assert has_invalid_param_error, f"Expected error about invalid parameter. Got: {error_messages}"

        # Check if suggestions are provided (may not be if name is too different)
        has_suggestions = any('Did you mean' in msg for msg in error_messages)
        if has_suggestions:
            print("✓ Fuzzy-matched suggestions provided")

        print("✓ Invalid parameter name correctly detected")
        print(f"✓ Error message: {error_messages[0]}")

    finally:
        # Clean up
        Path(config_path).unlink()


def test_multiple_invalid_options():
    """Test that multiple invalid parameters are all caught."""

    config_dict = {
        "name": "MultipleInvalidTest",
        "vplanet": {
            "inpath": FIXTURES_PATH,
            "executable": VPLANET_EXECUTABLE,
            "vplfile": "vpl.in"
        },
        "parameters": [
            {
                "name": "star.dBogusParam1",
                "bounds": [0.0, 1.0],
                "units": "dimensionless"
            },
            {
                "name": "star.dBogusParam2",
                "bounds": [0.0, 1.0],
                "units": "dimensionless"
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

        # Should catch both invalid parameters
        assert len(errors) >= 2, f"Expected at least 2 errors, got {len(errors)}"

        error_messages = [err[0] for err in errors]
        has_param1_error = any('dBogusParam1' in msg or 'Invalid VPlanet parameter' in msg
                              for msg in error_messages)
        has_param2_error = any('dBogusParam2' in msg or 'Invalid VPlanet parameter' in msg
                              for msg in error_messages)

        assert has_param1_error, "Should catch first invalid parameter"
        assert has_param2_error, "Should catch second invalid parameter"

        print(f"✓ Both invalid parameters detected ({len(errors)} errors total)")

    finally:
        Path(config_path).unlink()


def test_typo_gets_suggestions():
    """Test that a typo in a parameter name gets helpful suggestions."""

    config_dict = {
        "name": "TypoTest",
        "vplanet": {
            "inpath": FIXTURES_PATH,
            "executable": VPLANET_EXECUTABLE,
            "vplfile": "vpl.in"
        },
        "parameters": [
            {
                "name": "star.dMas",  # Typo: missing 's'
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

        assert len(errors) > 0, "Expected validation error for typo 'dMas'"

        error_messages = [err[0] for err in errors]

        # Should have suggestions including 'dMass'
        has_suggestions = any('Did you mean' in msg for msg in error_messages)
        assert has_suggestions, f"Expected suggestions for typo 'dMas'. Got: {error_messages}"

        has_dmass_suggestion = any('dMass' in msg for msg in error_messages)
        assert has_dmass_suggestion, f"Expected 'dMass' in suggestions. Got: {error_messages}"

        print("✓ Typo 'dMas' correctly detected")
        print(f"✓ Suggestions provided including 'dMass'")
        print(f"✓ Error message: {error_messages[0]}")

    finally:
        Path(config_path).unlink()


def test_valid_option_passes():
    """Test that valid parameter names pass validation."""

    config_dict = {
        "name": "ValidOptionTest",
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

        # Filter out errors that aren't about parameter names
        param_errors = [err for err in errors if 'dMass' in err[0] or 'Invalid VPlanet parameter' in err[0]]

        assert len(param_errors) == 0, f"Valid parameter 'dMass' should not raise errors. Got: {param_errors}"

        print("✓ Valid parameter 'dMass' passes validation")

    finally:
        Path(config_path).unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
