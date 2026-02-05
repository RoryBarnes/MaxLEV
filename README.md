# MaxLEV - Maximum Likelihood Estimation for VPLanet

A tool for finding maximum likelihood parameters of VPLanet simulations given observational constraints.

## Features

- **JSON Configuration**: Define optimization problems with parameters, outputs, and observables
- **VPLanet Validation**: Validates all parameters and outputs against VPLanet before running
- **Asymmetric Uncertainties**: Supports asymmetric Gaussian likelihoods
- **Parallelization**: Differential evolution with multiprocessing support
- **MaxLEV Input Files**: Generates `*_maxlev.in` files with best-fit parameter values

## Quick Start

```bash
# Run optimization
python maxlev.py examples/EarthInterior/earthInterior.json

# Validate configuration only
python maxlev.py examples/EarthInterior/earthInterior.json --validate
```

## Installation

```bash
git clone https://github.com/RoryBarnes/MaxLEV.git
pip install numpy scipy astropy vplanet vplanet_inference
```

## Documentation

Full documentation is available in the [docs/](docs/) folder:

- [Installation Guide](docs/install.rst)
- [Running MaxLEV](docs/help.rst)
- [Configuration Format](docs/config.rst)
- [Examples](docs/examples.rst)

## Example: EarthInterior

The `examples/EarthInterior/` directory contains a complete example that constrains Earth's initial thermal state using the `thermint` module:

- 6 optimization parameters (mantle temperature, CMB temperature, etc.)
- 11 observational constraints (heat flow, magnetic moment, etc.)
- Results: chi^2 = 6.08 with all observables within 1.5 sigma

See the [examples documentation](docs/examples.rst) for details.

## Output

After optimization, MaxLEV generates:

1. **Results file**: Best-fit parameters, chi^2, and residuals
2. **MaxLEV input files**: `*_maxlev.in` files with optimized values

Example output:
```
examples/EarthInterior/
    earth.in           # Original template
    earth_maxlev.in    # Generated with ML values
earthInterior_results.txt
```

## References

- Uses [vplanet_inference](https://github.com/VirtualPlanetaryLaboratory/vplanet_inference) for VPLanet integration
- Part of the [VPLanet](https://github.com/VirtualPlanetaryLaboratory/vplanet) ecosystem
