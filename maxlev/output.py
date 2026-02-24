"""Output generation for MaxLEV."""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List


def fnGenerateMaxLEVInputFiles(daBestParams: np.ndarray, config) -> List[str]:
    """
    Generate *_maxlev.in files with maximum likelihood parameter values.

    Args:
        daBestParams: Array of best-fit parameter values
        config: MaxLEVConfig object

    Returns:
        List of paths to generated files
    """
    sInpath = config.vplanet.get('inpath', '.')
    listGeneratedFiles = []

    # Group parameters by file
    dictParamsByFile = {}
    for i, param in enumerate(config.parameters):
        sFileName = param.file_name
        if sFileName not in dictParamsByFile:
            dictParamsByFile[sFileName] = []
        dictParamsByFile[sFileName].append((param.param_name, daBestParams[i]))

    # Process each file that has parameters to update
    for sFileName, listParams in dictParamsByFile.items():
        sInputFilePath = Path(sInpath) / f"{sFileName}.in"
        sOutputFilePath = Path(sInpath) / f"{sFileName}_maxlev.in"

        if not sInputFilePath.exists():
            print(f"  Warning: Template file not found: {sInputFilePath}")
            continue

        listOutputLines = _flistUpdateInputFile(sInputFilePath, listParams)
        with open(sOutputFilePath, 'w') as f:
            f.writelines(listOutputLines)

        listGeneratedFiles.append(str(sOutputFilePath))
        print(f"  [OK] Generated: {sOutputFilePath}")

    return listGeneratedFiles


def _flistUpdateInputFile(pathInputFile: Path,
                          listParams: List[tuple]) -> List[str]:
    """
    Read input file and update parameter values.

    Args:
        pathInputFile: Path to template .in file
        listParams: List of (param_name, value) tuples

    Returns:
        List of lines with updated values
    """
    with open(pathInputFile, 'r') as f:
        listLines = f.readlines()

    # Build dict for quick lookup
    dictParamValues = {sName: dValue for sName, dValue in listParams}

    listOutputLines = []
    for sLine in listLines:
        sUpdatedLine = _fsUpdateLine(sLine, dictParamValues)
        listOutputLines.append(sUpdatedLine)

    return listOutputLines


def _fsUpdateLine(sLine: str, dictParamValues: Dict[str, float]) -> str:
    """
    Update a single line if it contains a parameter to replace.

    Args:
        sLine: Original line from input file
        dictParamValues: Dict mapping parameter names to new values

    Returns:
        Updated line (or original if no match)
    """
    sStripped = sLine.strip()

    # Skip empty lines and comments
    if not sStripped or sStripped.startswith('#'):
        return sLine

    # Parse the line: parameter_name value [# comment]
    # VPLanet format: dTMan 3000 # comment
    listParts = sStripped.split()
    if len(listParts) < 2:
        return sLine

    sParamName = listParts[0]

    # Check if this parameter should be updated
    if sParamName not in dictParamValues:
        return sLine

    dNewValue = dictParamValues[sParamName]

    # Preserve the original line structure (whitespace, comments)
    # Find where the parameter name ends and value begins
    matchParam = re.match(r'^(\s*)(\S+)(\s+)(\S+)(.*)', sLine)
    if not matchParam:
        return sLine

    sLeadingWhitespace = matchParam.group(1)
    sOrigParamName = matchParam.group(2)
    sSeparator = matchParam.group(3)
    sRemainder = matchParam.group(5)  # Everything after value (comments, etc.)

    # Format the new value appropriately
    if abs(dNewValue) >= 1e4 or (abs(dNewValue) < 1e-3 and dNewValue != 0):
        sNewValue = f"{dNewValue:.6e}"
    else:
        sNewValue = f"{dNewValue:.6f}".rstrip('0').rstrip('.')

    return f"{sLeadingWhitespace}{sOrigParamName}{sSeparator}{sNewValue}{sRemainder}\n"


def save_results(best_params: np.ndarray, best_value: float,
                 config, model, output_settings: Dict[str, Any],
                 prior_collection=None):
    """Save optimization results to file."""

    filepath = output_settings.get('results_file', 'maxlike_results.txt')
    bHasPriors = prior_collection is not None and prior_collection.fbHasPriors()

    with open(filepath, 'w') as f:
        if bHasPriors:
            f.write(f"{config.name} Maximum A Posteriori (MAP) Estimation\n")
        else:
            f.write(f"{config.name} Maximum Likelihood Estimation\n")
        f.write("=" * 70 + "\n\n")

        if bHasPriors:
            f.write("MAP Parameters:\n")
        else:
            f.write("Maximum Likelihood Parameters:\n")
        f.write("-" * 70 + "\n")
        for i, param in enumerate(config.parameters):
            f.write(f"{param.name:30s} = {best_params[i]:.6e}\n")

        if bHasPriors:
            dNegLogLike = model.neg_log_likelihood(best_params)
            dNegLogPrior = prior_collection.fdNegLogPrior(best_params)
            f.write(f"\n-ln(Posterior)  = {best_value:.6e}\n")
            f.write(f"-ln(Likelihood) = {dNegLogLike:.6e}\n")
            f.write(f"-ln(Prior)      = {dNegLogPrior:.6e}\n")
            f.write(f"chi^2           = {2*dNegLogLike:.6e}\n")
        else:
            f.write(f"\n-ln(Likelihood) = {best_value:.6e}\n")
            f.write(f"chi^2           = {2*best_value:.6e}\n")

        # Model predictions at maximum likelihood
        try:
            outputs = model.run_simulation(best_params)
            if outputs is not None:
                computed = model.observable_computer.compute(outputs)

                f.write("\nModel Predictions:\n")
                f.write("-" * 70 + "\n")
                for obs in config.observables:
                    model_val = computed[obs.name]
                    obs_val = obs.observed_value
                    sigma = obs.get_uncertainty(model_val)
                    residual = (model_val - obs_val) / sigma
                    f.write(f"{obs.name:30s} = {model_val:.6e} (obs: {obs_val:.6e})\n")
                    f.write(f"  Residual: {residual:+.3f} sigma\n")
        except Exception as e:
            f.write(f"\nCould not compute model predictions: {e}\n")

        f.write("\nParameter Bounds:\n")
        f.write("-" * 70 + "\n")
        for param in config.parameters:
            f.write(f"{param.name:30s}: [{param.bounds[0]:.6e}, {param.bounds[1]:.6e}]\n")

    print(f"\n[OK] Results saved: {filepath}")

    # Generate *_maxlev.in files
    print("\nGenerating MaxLEV input files...")
    fnGenerateMaxLEVInputFiles(best_params, config)


def plot_evolution(best_params: np.ndarray, config, model,
                   output_settings: Dict[str, Any]):
    """Generate evolution plot if configured."""

    if not output_settings.get('plot_evolution', False):
        return

    print("\nGenerating evolution plot...")

    evol_config = output_settings.get('evolution', {})
    time_range = evol_config.get('time_range', [0.1, 13.0])
    num_points = evol_config.get('num_points', 100)
    log_scale = evol_config.get('log_scale', True)

    # Generate time grid
    if log_scale:
        times = np.logspace(np.log10(time_range[0]),
                           np.log10(time_range[1]),
                           num_points)
    else:
        times = np.linspace(time_range[0], time_range[1], num_points)

    # Find the stop time parameter index
    stop_time_idx = None
    for i, param in enumerate(config.parameters):
        param_lower = param.name.lower()
        if 'stoptime' in param_lower or param_lower.endswith('.dstoptime'):
            stop_time_idx = i
            break

    if stop_time_idx is None:
        print("  Warning: Could not identify stop time parameter for evolution plot")
        return

    # Compute evolution
    computed_evol = {obs.name: np.zeros(num_points) for obs in config.observables}

    for i, t in enumerate(times):
        theta = best_params.copy()
        theta[stop_time_idx] = t

        outputs = model.run_simulation(theta)
        if outputs is not None:
            try:
                computed = model.observable_computer.compute(outputs)
                for obs_name in computed_evol:
                    computed_evol[obs_name][i] = computed[obs_name]
            except:
                for obs_name in computed_evol:
                    computed_evol[obs_name][i] = np.nan
        else:
            for obs_name in computed_evol:
                computed_evol[obs_name][i] = np.nan

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    for obs in config.observables:
        ax.plot(times, computed_evol[obs.name], '-', linewidth=2,
               label=f'{obs.name} (model)')

        # Add observed value marker at best-fit time
        age = best_params[stop_time_idx]
        obs_val = obs.observed_value
        ax.scatter([age], [obs_val], s=100, zorder=10,
                  label=f'{obs.name} (observed)')

    ax.set_xlabel('Time [Gyr]', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_title(f'{config.name}: Evolution', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.tight_layout()

    plot_file = output_settings.get('plot_file', 'evolution.pdf')
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved: {plot_file}")
