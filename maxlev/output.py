"""Output generation for MaxLEV."""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List


def _flistFileNamesForParameter(param) -> List[str]:
    """Return all body file names a parameter writes to."""
    if param.bIsShared:
        return list(param.bodies)
    return [param.sFileName]


def _dictGroupParamsByFile(daBestParams, config) -> Dict[str, List[tuple]]:
    """Group parameter values by the body file they belong to."""
    dictParamsByFile = {}
    for i, param in enumerate(config.parameters):
        for sFileName in _flistFileNamesForParameter(param):
            if sFileName not in dictParamsByFile:
                dictParamsByFile[sFileName] = []
            dictParamsByFile[sFileName].append(
                (param.sParamName, daBestParams[i])
            )
    return dictParamsByFile


def _fnWriteMaxLEVFile(sInpath, sFileName, listParams):
    """Write a single *_maxlev.in file, returning its path or None."""
    sInputFilePath = Path(sInpath) / f"{sFileName}.in"
    sOutputFilePath = Path(sInpath) / f"{sFileName}_maxlev.in"

    if not sInputFilePath.exists():
        print(f"  Warning: Template file not found: {sInputFilePath}")
        return None

    listOutputLines = _flistUpdateInputFile(sInputFilePath, listParams)
    with open(sOutputFilePath, 'w') as f:
        f.writelines(listOutputLines)

    print(f"  [OK] Generated: {sOutputFilePath}")
    return str(sOutputFilePath)


def fnGenerateMaxLEVInputFiles(daBestParams: np.ndarray,
                               config) -> List[str]:
    """
    Generate *_maxlev.in files with best-fit parameter values.

    Args:
        daBestParams: Array of best-fit parameter values
        config: MaxLEVConfig object

    Returns:
        List of paths to generated files
    """
    sInpath = config.vplanet.get('inpath', '.')
    dictParamsByFile = _dictGroupParamsByFile(daBestParams, config)

    listGeneratedFiles = []
    for sFileName, listParams in dictParamsByFile.items():
        sPath = _fnWriteMaxLEVFile(sInpath, sFileName, listParams)
        if sPath is not None:
            listGeneratedFiles.append(sPath)
    return listGeneratedFiles


def _flistUpdateInputFile(pathInputFile: Path,
                          listParams: List[tuple]) -> List[str]:
    """Read input file and replace parameter values."""
    with open(pathInputFile, 'r') as f:
        listLines = f.readlines()
    dictParamValues = {sName: dValue for sName, dValue in listParams}
    return [_fsUpdateLine(sLine, dictParamValues) for sLine in listLines]


def _fsFormatValue(dValue: float) -> str:
    """Format a numeric value for VPLanet input files."""
    if abs(dValue) >= 1e4 or (abs(dValue) < 1e-3 and dValue != 0):
        return f"{dValue:.6e}"
    return f"{dValue:.6f}".rstrip('0').rstrip('.')


def _fsUpdateLine(sLine: str, dictParamValues: Dict[str, float]) -> str:
    """Update a single line if it contains a parameter to replace."""
    sStripped = sLine.strip()
    if not sStripped or sStripped.startswith('#'):
        return sLine

    listParts = sStripped.split()
    if len(listParts) < 2:
        return sLine

    sParamName = listParts[0]
    if sParamName not in dictParamValues:
        return sLine

    return _fsReplacedLine(sLine, sParamName, dictParamValues[sParamName])


def _fsReplacedLine(sLine, sParamName, dNewValue):
    """Rebuild a VPLanet input line with a new parameter value."""
    matchParam = re.match(r'^(\s*)(\S+)(\s+)(\S+)(.*)', sLine)
    if not matchParam:
        return sLine
    sLeading = matchParam.group(1)
    sOrigName = matchParam.group(2)
    sSeparator = matchParam.group(3)
    sRemainder = matchParam.group(5)
    sNewValue = _fsFormatValue(dNewValue)
    return f"{sLeading}{sOrigName}{sSeparator}{sNewValue}{sRemainder}\n"


def fnSaveResults(daBestParams: np.ndarray, dBestValue: float,
                  config, model, dictOutputSettings: Dict[str, Any],
                  prior_collection=None):
    """Save optimization results to file."""
    sFilepath = dictOutputSettings.get('results_file',
                                       'maxlike_results.txt')
    bHasPriors = (prior_collection is not None
                  and prior_collection.fbHasPriors())

    with open(sFilepath, 'w') as f:
        _fnWriteHeader(f, config.name, bHasPriors)
        _fnWriteParameters(f, config.parameters, daBestParams, bHasPriors)
        _fnWriteStatistics(f, dBestValue, bHasPriors, model,
                           daBestParams, prior_collection)
        _fnWritePredictions(f, model, config, daBestParams)
        _fnWriteBounds(f, config.parameters)

    print(f"\n[OK] Results saved: {sFilepath}")
    print("\nGenerating MaxLEV input files...")
    fnGenerateMaxLEVInputFiles(daBestParams, config)


def _fnWriteHeader(f, sName, bHasPriors):
    """Write the results file header."""
    if bHasPriors:
        f.write(f"{sName} Maximum A Posteriori (MAP) Estimation\n")
    else:
        f.write(f"{sName} Maximum Likelihood Estimation\n")
    f.write("=" * 70 + "\n\n")


def _fnWriteParameters(f, listParams, daBestParams, bHasPriors):
    """Write the parameter values section."""
    if bHasPriors:
        f.write("MAP Parameters:\n")
    else:
        f.write("Maximum Likelihood Parameters:\n")
    f.write("-" * 70 + "\n")
    for i, param in enumerate(listParams):
        sTag = param.fsSharedTag()
        f.write(f"{param.name:30s} = {daBestParams[i]:.6e}{sTag}\n")


def _fnWriteStatistics(f, dBestValue, bHasPriors, model,
                       daBestParams, priorCollection):
    """Write likelihood/posterior statistics."""
    if bHasPriors:
        dNegLogLike = model.fdNegLogLikelihood(daBestParams)
        dNegLogPrior = priorCollection.fdNegLogPrior(daBestParams)
        f.write(f"\n-ln(Posterior)  = {dBestValue:.6e}\n")
        f.write(f"-ln(Likelihood) = {dNegLogLike:.6e}\n")
        f.write(f"-ln(Prior)      = {dNegLogPrior:.6e}\n")
        f.write(f"chi^2           = {2*dNegLogLike:.6e}\n")
    else:
        f.write(f"\n-ln(Likelihood) = {dBestValue:.6e}\n")
        f.write(f"chi^2           = {2*dBestValue:.6e}\n")


def _fnWritePredictions(f, model, config, daBestParams):
    """Write model predictions vs observations."""
    try:
        daOutputs = model.fdaRunSimulation(daBestParams)
        if daOutputs is None:
            return
        dictComputed = model.observable_computer.compute(daOutputs)
        f.write("\nModel Predictions:\n")
        f.write("-" * 70 + "\n")
        for obs in config.observables:
            dModelVal = dictComputed[obs.name]
            dObsVal = obs.observed_value
            dSigma = obs.fdGetUncertainty(dModelVal)
            dResidual = (dModelVal - dObsVal) / dSigma
            f.write(f"{obs.name:30s} = {dModelVal:.6e} "
                    f"(obs: {dObsVal:.6e})\n")
            f.write(f"  Residual: {dResidual:+.3f} sigma\n")
    except Exception as e:
        f.write(f"\nCould not compute model predictions: {e}\n")


def _fnWriteBounds(f, listParams):
    """Write the parameter bounds section."""
    f.write("\nParameter Bounds:\n")
    f.write("-" * 70 + "\n")
    for param in listParams:
        f.write(f"{param.name:30s}: "
                f"[{param.bounds[0]:.6e}, {param.bounds[1]:.6e}]\n")


def fnPlotEvolution(daBestParams: np.ndarray, config, model,
                    dictOutputSettings: Dict[str, Any]):
    """Generate evolution plot if configured."""
    if not dictOutputSettings.get('plot_evolution', False):
        return
    print("\nGenerating evolution plot...")

    dictEvol = dictOutputSettings.get('evolution', {})
    iStopTimeIdx = _fiFindStopTimeIndex(config.parameters)
    if iStopTimeIdx is None:
        print("  Warning: Could not identify stop time parameter "
              "for evolution plot")
        return

    daTimes = _fdaGenerateTimeGrid(dictEvol)
    dictComputedEvol = _fdictComputeEvolution(
        daTimes, daBestParams, iStopTimeIdx, model, config,
    )
    _fnRenderPlot(
        daTimes, dictComputedEvol, daBestParams, iStopTimeIdx,
        config, dictEvol, dictOutputSettings,
    )


def _fiFindStopTimeIndex(listParams) -> int:
    """Return index of the stop-time parameter, or None."""
    for i, param in enumerate(listParams):
        sLower = param.name.lower()
        if 'stoptime' in sLower or sLower.endswith('.dstoptime'):
            return i
    return None


def _fdaGenerateTimeGrid(dictEvol) -> np.ndarray:
    """Build a time array from evolution config."""
    listTimeRange = dictEvol.get('time_range', [0.1, 13.0])
    iNumPoints = dictEvol.get('num_points', 100)
    bLogScale = dictEvol.get('log_scale', True)
    if bLogScale:
        return np.logspace(np.log10(listTimeRange[0]),
                           np.log10(listTimeRange[1]),
                           iNumPoints)
    return np.linspace(listTimeRange[0], listTimeRange[1], iNumPoints)


def _fdictComputeEvolution(daTimes, daBestParams, iStopTimeIdx,
                           model, config):
    """Run simulations at each time point and collect observables."""
    dictEvol = {obs.name: np.zeros(len(daTimes))
                for obs in config.observables}
    for i, dTime in enumerate(daTimes):
        daTheta = daBestParams.copy()
        daTheta[iStopTimeIdx] = dTime
        _fnFillEvolutionPoint(i, daTheta, model, dictEvol)
    return dictEvol


def _fnFillEvolutionPoint(i, daTheta, model, dictEvol):
    """Compute observables for a single time point."""
    daOutputs = model.fdaRunSimulation(daTheta)
    if daOutputs is None:
        for sKey in dictEvol:
            dictEvol[sKey][i] = np.nan
        return
    try:
        dictComputed = model.observable_computer.compute(daOutputs)
        for sKey in dictEvol:
            dictEvol[sKey][i] = dictComputed[sKey]
    except Exception:
        for sKey in dictEvol:
            dictEvol[sKey][i] = np.nan


def _fnRenderPlot(daTimes, dictComputedEvol, daBestParams,
                  iStopTimeIdx, config, dictEvol, dictOutputSettings):
    """Create and save the evolution figure."""
    fig, ax = plt.subplots(figsize=(10, 7))
    for obs in config.observables:
        ax.plot(daTimes, dictComputedEvol[obs.name], '-',
                linewidth=2, label=f'{obs.name} (model)')
        dAge = daBestParams[iStopTimeIdx]
        ax.scatter([dAge], [obs.observed_value], s=100, zorder=10,
                   label=f'{obs.name} (observed)')

    ax.set_xlabel('Time [Gyr]', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    ax.set_title(f'{config.name}: Evolution', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    if dictEvol.get('log_scale', True):
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.tight_layout()
    sPlotFile = dictOutputSettings.get('plot_file', 'evolution.pdf')
    fig.savefig(sPlotFile, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved: {sPlotFile}")
