# vmle - VPlanet Maximum Likelihood Estimation

A general-purpose MLE tool for VPlanet stellar evolution models with comprehensive input validation.

## Features

- **JSON Configuration**: Easy-to-use JSON format for defining optimization problems
- **VPlanet Validation**: Validates all parameters, outputs, and body names against VPlanet's actual options
- **Asymmetric Uncertainties**: Supports asymmetric Gaussian likelihoods with separate lower/upper error bars
- **Parallelization**: Differential evolution with multiprocessing support
- **Extensible**: Works with any VPlanet problem, not just specific systems

## Installation

The package requires:
- Python 3.7+
- numpy
- scipy
- matplotlib
- astropy
- vplanet_inference

```bash
cd /Users/rory/src/vmle
# No installation needed - run directly with python vmle.py
```

## Usage

### Basic Usage

```bash
python vmle.py config.json
```

### Validation Only

```bash
python vmle.py config.json --validate
```

### With Options

```bash
python vmle.py config.json --workers 4 --maxiter 2000 --seed 123
```

### Command-Line Options

- `--validate`: Validate configuration without running optimization
- `--verbose, -v`: Enable verbose VPlanet output
- `--seed N`: Override random seed
- `--maxiter N`: Override maximum iterations
- `--workers N`: Override number of parallel workers (1 = serial, -1 = all CPUs)
- `--output-dir PATH`: Override output directory

## Configuration File Format

See [gj1132_ribas_test.json](gj1132_ribas_test.json:1-0) for a complete example.

### Key Sections

#### 1. VPlanet Settings

```json
{
  "vplanet": {
    "inpath": "/path/to/vplanet/input/files",
    "executable": "/path/to/vplanet/bin/vplanet",
    "vplfile": "vpl.in",
    "verbose": false
  }
}
```

#### 2. Parameters to Optimize

```json
{
  "parameters": [
    {
      "name": "star.dMass",
      "bounds": [0.17, 0.22],
      "units": "Msun",
      "description": "Stellar mass"
    }
  ]
}
```

**Important**:
- Parameter names are validated against VPlanet's `-h` output
- Format: `filename.parameter` (e.g., `star.dMass`, `vpl.dStopTime`)
- Units support: `Msun`, `Gyr`, `Lsun`, `dimensionless`, `dex(dimensionless)`, etc.

#### 3. Output Parameters

```json
{
  "outputs": [
    {
      "name": "final.star.Luminosity",
      "units": "Lsun"
    }
  ]
}
```

**Important**:
- Output names are validated against VPlanet's output parameters
- Body names are validated against `saBodyFiles` in vpl.in
- Outputs are validated against `saOutputOrder` in the body's `.in` file

#### 4. Observables

**Direct Observable** (uses output directly):
```json
{
  "name": "L_bol",
  "type": "direct",
  "output": "final.star.Luminosity",
  "observed_value": 4.38e-3,
  "uncertainty": 3.4e-4
}
```

**Derived Observable** (computed from expression):
```json
{
  "name": "log_Lxuv_Lbol",
  "type": "derived",
  "expression": "log10(final.star.LXUVStellar / final.star.Luminosity)",
  "observed_value": -4.26,
  "uncertainty": 0.15
}
```

**Asymmetric Uncertainties**:
```json
{
  "name": "some_observable",
  "type": "direct",
  "output": "final.star.SomeParameter",
  "observed_value": 1.0,
  "uncertainty_lower": 0.1,
  "uncertainty_upper": 0.2
}
```

#### 5. Optimizer Settings

```json
{
  "optimizer": {
    "algorithm": "differential_evolution",
    "seed": 42,
    "maxiter": 1000,
    "tol": 0.01,
    "de_settings": {
      "strategy": "best1bin",
      "popsize": 15,
      "mutation": [0.5, 1.0],
      "recombination": 0.7,
      "workers": 1,
      "updating": "deferred",
      "polish": false,
      "disp": true
    }
  }
}
```

**Note**: `polish: false` is critical to enforce parameter bounds.

## Validation Features

vmle performs comprehensive validation:

1. **Parameter Name Validation**: Checks against VPlanet's valid input options
2. **Output Parameter Validation**: Checks against VPlanet's valid output parameters
3. **Body Name Validation**: Verifies body names exist in `saBodyFiles`
4. **saOutputOrder Validation**: Ensures requested outputs are in the body's output list
5. **Expression Validation**: Validates derived observable expressions reference valid outputs

Validation errors include:
- Line numbers in the JSON file
- Fuzzy-matched suggestions for typos
- Clear error messages with context

## Example: GJ 1132 Ribas XUV Model

To reproduce MaxLikelihoodRibas.py results:

```bash
python vmle.py gj1132_ribas_test.json
```

This will:
1. Validate all parameters and outputs against VPlanet
2. Run differential evolution optimization (this takes many hours!)
3. Save results to `maxlike_results.txt`
4. Generate evolution plot `gj1132_maxlike_evolution.pdf`

## Parallelization

For faster optimization:

```bash
python vmle.py config.json --workers 4
```

**Note**: With `workers > 1`, results will not be exactly reproducible even with the same seed, but should converge to similar values.

## Troubleshooting

### "Invalid VPlanet parameter 'XXX'"

The parameter name is not recognized by VPlanet. Check:
- Correct spelling (use suggestions from error message)
- Parameter exists in VPlanet version you're using
- Run `vplanet -h` to see all valid options

### "Output 'XXX' not in saOutputOrder for body 'YYY'"

The output parameter is not in the body's `saOutputOrder` line. Either:
- Add it to `saOutputOrder` in the body's `.in` file
- Use a different output parameter that's already listed

### "Body 'XXX' not found in vpl.in"

Check `saBodyFiles` in your `vpl.in` file to see valid body names.

## Implementation Details

- **Alphabetical Sorting**: VPlanet outputs are sorted alphabetically by vplanet_inference
- **Unit Handling**: Supports `dex` units for log10-space parameters
- **Thread Safety**: Uses vplanet_inference's random temp directories for parallel runs
- **Error Handling**: Returns penalty value (1e10) for failed VPlanet runs

## Files Created

- `validation.py`: VPlanet option validation
- `config.py`: JSON configuration loading
- `units.py`: Unit string parsing
- `likelihood.py`: Gaussian and asymmetric Gaussian likelihoods
- `observables.py`: Observable computation
- `model.py`: VplanetModel wrapper
- `optimizer.py`: Optimization algorithms
- `output.py`: Results and plotting
- `cli.py`: Command-line interface
- `vmle.py`: Main entry point

## References

- Based on MaxLikelihoodRibas.py for GJ 1132
- Uses vplanet_inference for VPlanet integration
- Follows vplot patterns for VPlanet option validation
