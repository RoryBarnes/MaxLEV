# Test Fixtures

Minimal VPlanet input files for testing MaxLEV.

## Files

### vpl.in
Main VPlanet configuration file with basic system setup.

**Purpose**: Defines a minimal test system for validation tests.

**Contents**:
- System name: `test_system`
- One body file: `star.in`
- Basic time integration settings

### star.in
Stellar body configuration with stellar evolution module.

**Purpose**: Provides a valid body file with saOutputOrder for validation tests.

**Contents**:
- Stellar parameters (mass, rotation, age)
- Stellar evolution model: Baraffe
- XUV evolution parameters
- **saOutputOrder**: `Time -Luminosity -LXUVStellar -Radius Temperature`

## Usage in Tests

These fixtures are used by all validation tests via `test_utils.py`:

```python
from tests.test_utils import FIXTURES_PATH, VPLANET_EXECUTABLE

config_dict = {
    "vplanet": {
        "inpath": FIXTURES_PATH,
        "executable": VPLANET_EXECUTABLE,
        "vplfile": "vpl.in"
    },
    ...
}
```

## Why These Files?

1. **Self-contained**: Tests don't depend on external data
2. **Minimal**: Only essential parameters for testing
3. **Portable**: Work on any system with VPlanet installed
4. **Realistic**: Mirror actual VPlanet input file structure

## saOutputOrder Validation

The `star.in` file includes `saOutputOrder` to test:
- Valid outputs (Luminosity, LXUVStellar) ✓
- Invalid outputs (Mass - not in list) ✗
- Observable expressions referencing outputs

This is critical for validation tests to verify MaxLEV correctly checks that requested outputs will actually be printed by VPlanet.

## Modification

If you add new validation tests that require different outputs:
1. Update `saOutputOrder` in `star.in`
2. Ensure outputs are compatible with the `stellar` module
3. Document changes in this README
