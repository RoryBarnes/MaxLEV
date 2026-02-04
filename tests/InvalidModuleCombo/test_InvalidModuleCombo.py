"""Test invalid combination of output parameter and modules."""

import pytest
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.config import MaxLEVConfig
from tests.test_utils import FIXTURES_PATH, VPLANET_EXECUTABLE, FIXTURES_DIR


def test_output_not_in_saOutputOrder():
    """
    Test that outputs not in the body's saOutputOrder are caught.

    This is the most critical module compatibility check - if the output
    isn't in saOutputOrder, VPlanet won't print it.
    """

    config_dict = {
        "name": "MissingOutputOrderTest",
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
            # Request an output that exists in VPlanet but is NOT in star.in's saOutputOrder
            {
                "name": "final.star.Mass",  # Valid parameter, but not in saOutputOrder
                "units": "Msun"
            }
        ],
        "observables": [
            {
                "name": "test_obs",
                "type": "direct",
                "output": "final.star.Mass",
                "observed_value": 0.2,
                "uncertainty": 0.01
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

        # Should have error about Mass not in saOutputOrder
        assert len(errors) > 0, "Expected validation error for output not in saOutputOrder"

        error_messages = [err[0] for err in errors]
        has_output_order_error = any('saOutputOrder' in msg and 'Mass' in msg
                                     for msg in error_messages)

        assert has_output_order_error, f"Expected error about Mass not in saOutputOrder. Got: {error_messages}"

        print("✓ Missing output in saOutputOrder correctly detected")
        print(f"✓ Error message: {error_messages[0]}")

    finally:
        Path(config_path).unlink()


def test_valid_outputs_in_saOutputOrder():
    """Test that outputs actually in saOutputOrder pass validation."""

    config_dict = {
        "name": "ValidOutputOrderTest",
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
            # These SHOULD be in star.in's saOutputOrder based on our fixture
            {
                "name": "final.star.Luminosity",
                "units": "Lsun"
            },
            {
                "name": "final.star.LXUVStellar",
                "units": "Lsun"
            }
        ],
        "observables": [
            {
                "name": "test_obs",
                "type": "direct",
                "output": "final.star.Luminosity",
                "observed_value": 0.004,
                "uncertainty": 0.0003
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

        # Should not have saOutputOrder errors for these outputs
        output_order_errors = [err for err in errors
                              if 'saOutputOrder' in err[0] and
                              ('Luminosity' in err[0] or 'LXUVStellar' in err[0])]

        assert len(output_order_errors) == 0, \
            f"Valid outputs in saOutputOrder should pass. Got: {output_order_errors}"

        print("✓ Outputs in saOutputOrder pass validation")

    finally:
        Path(config_path).unlink()


def test_check_saOutputOrder_file():
    """
    Verify we can read the actual saOutputOrder from the test fixture.

    This is a sanity check to understand what's actually in the file.
    """
    star_in_path = FIXTURES_DIR / "star.in"

    if not star_in_path.exists():
        pytest.skip("Test fixture star.in not found")

    with open(star_in_path, 'r') as f:
        content = f.read()

    # Find saOutputOrder line
    for line in content.split('\n'):
        if line.strip().startswith('saOutputOrder'):
            print(f"✓ Found saOutputOrder in star.in:")
            print(f"  {line.strip()}")

            # Parse the outputs
            parts = line.split()
            outputs = [p.lstrip('-') for p in parts[1:] if p != '$']
            print(f"✓ Outputs in saOutputOrder: {outputs}")

            # Check expected outputs
            assert 'Luminosity' in outputs or 'Time' in outputs, \
                "Expected to find Luminosity or Time in saOutputOrder"
            break
    else:
        pytest.fail("Could not find saOutputOrder in star.in")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
