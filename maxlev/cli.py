"""Command-line interface for MaxLEV."""

import argparse
import sys
from pathlib import Path


def fParseArgs():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='VPlanet Maximum Likelihood Estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  maxlev config.json              Run MLE with JSON config
  maxlev config.json --validate   Validate config without running
  maxlev config.json --verbose    Enable verbose output
  maxlev config.json --workers 4  Use 4 parallel workers
        """
    )
    _fnAddArguments(parser)
    return parser.parse_args()


def _fnAddArguments(parser):
    """Register all CLI arguments on the parser."""
    parser.add_argument('config', type=str,
                        help='Path to configuration file (JSON)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate configuration without running')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed from config')
    parser.add_argument('--maxiter', type=int, default=None,
                        help='Override maximum iterations from config')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--workers', type=int, default=None,
                        help='Override number of parallel workers')


def main():
    """Main entry point for MaxLEV CLI."""
    args = fParseArgs()
    config = _fLoadConfig(args)
    _fnApplyOverrides(args, config)
    _fnValidateConfig(config, args.validate)
    _fnRunOptimization(config)


def _fLoadConfig(args):
    """Load and validate config file path, returning MaxLEVConfig."""
    from .config import MaxLEVConfig

    pathConfig = Path(args.config)
    if not pathConfig.exists():
        print(f"Error: Configuration file not found: {pathConfig}")
        sys.exit(1)

    print("=" * 70)
    print("VPlanet Maximum Likelihood Estimation")
    print("=" * 70)
    print(f"\nLoading configuration: {pathConfig}")

    if not pathConfig.suffix == '.json':
        print("Error: Configuration file must be JSON format")
        sys.exit(1)

    config = MaxLEVConfig.from_json(str(pathConfig))
    _fnPrintConfigSummary(config)
    return config


def _fnPrintConfigSummary(config):
    """Print parameter and observable counts."""
    print(f"  Name: {config.name}")
    iSharedCount = sum(1 for p in config.parameters if p.bIsShared)
    iExpandedCount = sum(
        len(p.flistExpandedNames()) for p in config.parameters
    )
    if iSharedCount > 0:
        print(f"  Parameters: {len(config.parameters)} free "
              f"({iSharedCount} shared, {iExpandedCount} expanded)")
    else:
        print(f"  Parameters: {len(config.parameters)}")
    print(f"  Observables: {len(config.observables)}")


def _fnApplyOverrides(args, config):
    """Apply command-line overrides to config."""
    if args.seed is not None:
        config.optimizer['seed'] = args.seed
    if args.maxiter is not None:
        config.optimizer['maxiter'] = args.maxiter
    if args.output_dir is not None:
        config.output_settings['results_dir'] = args.output_dir
    if args.verbose:
        config.vplanet['verbose'] = True
    if args.workers is not None:
        if 'de_settings' not in config.optimizer:
            config.optimizer['de_settings'] = {}
        config.optimizer['de_settings']['workers'] = args.workers


def _fnValidateConfig(config, bValidateOnly):
    """Validate config and optionally exit if --validate."""
    print("\nValidating configuration...")
    listErrors = config.validate()
    if listErrors:
        _fnPrintValidationErrors(listErrors)
        sys.exit(1)
    print("[OK] Configuration validated")
    if bValidateOnly:
        print("\nValidation complete (--validate flag, "
              "not running optimization)")
        sys.exit(0)


def _fnPrintValidationErrors(listErrors):
    """Print validation errors and advice."""
    print("\n" + "=" * 70)
    print("CONFIGURATION ERRORS")
    print("=" * 70)
    for sMsg, iLine in listErrors:
        if iLine > 0:
            print(f"Line {iLine}: {sMsg}")
        else:
            print(sMsg)
    print("\nPlease fix the errors above and try again.")


def _fnRunOptimization(config):
    """Initialize components, run optimizer, and report results."""
    from .likelihood import create_likelihood
    from .observables import ObservableComputer
    from .model import MaxLEVModel, AllSimulationsFailedError
    from .optimizer import Optimizer
    from .prior import flistCreatePriors

    model, priorCollection = _fInitializeModel(config)
    objectiveFunction = _fSelectObjective(model, priorCollection)

    optimizer = Optimizer(config.optimizer)
    try:
        daBestParams, dBestValue, dictResultInfo = optimizer.optimize(
            objectiveFunction,
            model.bounds,
            model.listParamNames,
        )
    except AllSimulationsFailedError as e:
        _fnReportAbortedOptimization(e)
        sys.exit(1)

    if dBestValue >= model.dFailurePenalty:
        _fnReportFailedOptimization(dBestValue, model)
        sys.exit(1)

    _fnReportResults(
        daBestParams, dBestValue, config, model,
        config.output_settings, prior_collection=priorCollection,
    )


def _fInitializeModel(config):
    """Build likelihood, observable computer, priors, and model."""
    from .likelihood import create_likelihood
    from .observables import ObservableComputer
    from .model import MaxLEVModel
    from .prior import flistCreatePriors

    print("\nInitializing VPlanet model...")
    listOutputNames = [o.name for o in config.outputs]
    likelihoodModel = create_likelihood(
        config.likelihood, config.observables,
    )
    observableComputer = ObservableComputer(
        listOutputNames, config.observables,
    )
    listPriorConfigs = [
        p.prior if p.prior is not None else {"type": "uniform"}
        for p in config.parameters
    ]
    priorCollection = flistCreatePriors(listPriorConfigs)
    model = MaxLEVModel(
        config, likelihoodModel, observableComputer,
        prior_collection=priorCollection,
    )
    print("[OK] VPlanet model initialized")
    return model, priorCollection


def _fSelectObjective(model, priorCollection):
    """Choose MAP or MLE objective based on priors."""
    if priorCollection.fbHasPriors():
        print("[OK] Priors detected: optimizing posterior (MAP)")
        return model.fdNegLogPosterior
    print("[OK] No priors: optimizing likelihood (MLE)")
    return model.fdNegLogLikelihood


def _fnReportResults(daBestParams, dBestValue, config, model,
                     dictOutputSettings, prior_collection=None):
    """Print results, save files, and report success."""
    from .output import fnSaveResults, fnPlotEvolution

    bHasPriors = (prior_collection is not None
                  and prior_collection.fbHasPriors())

    print("\n" + "=" * 70)
    if bHasPriors:
        print("Maximum A Posteriori (MAP) Results")
    else:
        print("Maximum Likelihood Results")
    print("=" * 70)

    _fnPrintParameterValues(config.parameters, daBestParams)
    _fnPrintStatistics(dBestValue, bHasPriors, model,
                       daBestParams, prior_collection)

    fnSaveResults(daBestParams, dBestValue, config, model,
                  dictOutputSettings, prior_collection=prior_collection)
    fnPlotEvolution(daBestParams, config, model, dictOutputSettings)

    print("\n" + "=" * 70)
    if bHasPriors:
        print("MAP estimation completed successfully!")
    else:
        print("Maximum likelihood estimation completed successfully!")
    print("=" * 70 + "\n")


def _fnPrintParameterValues(listParams, daBestParams):
    """Print each parameter's best-fit value to console."""
    for i, param in enumerate(listParams):
        sTag = param.fsSharedTag()
        print(f"{param.name:30s} = {daBestParams[i]:.6e}{sTag}")


def _fnPrintStatistics(dBestValue, bHasPriors, model,
                       daBestParams, priorCollection):
    """Print likelihood/posterior statistics to console."""
    if bHasPriors:
        dNegLogLike = model.fdNegLogLikelihood(daBestParams)
        dNegLogPrior = priorCollection.fdNegLogPrior(daBestParams)
        print(f"\n-ln(Posterior)  = {dBestValue:.6e}")
        print(f"-ln(Likelihood) = {dNegLogLike:.6e}")
        print(f"-ln(Prior)      = {dNegLogPrior:.6e}")
        print(f"chi^2           = {2*dNegLogLike:.6e}")
    else:
        print(f"\n-ln(Likelihood) = {dBestValue:.6e}")
        print(f"chi^2           = {2*dBestValue:.6e}")


def _fnReportAbortedOptimization(error):
    """Report that optimization was aborted due to total failure."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION ABORTED")
    print("=" * 70)
    print(f"\n{error}")
    print("\nTry running 'vplanet vpl.in' in the input directory to")
    print("diagnose the problem.")


def _fnReportFailedOptimization(dBestValue, model):
    """Report that optimization finished but found no valid solution."""
    print("\n" + "=" * 70)
    print("OPTIMIZATION FAILED")
    print("=" * 70)
    print(f"\nBest objective value ({dBestValue:.6e}) equals the failure")
    print("penalty, meaning no VPLanet simulation produced valid output.")
    print(f"\nSimulations attempted: {model.iSimulationCount}")
    print(f"Simulations failed:   {model.iSimulationFailureCount}")
    print("\nPossible causes:")
    print("  - VPLanet executable is incompatible with the input files")
    print("  - All parameter combinations produce unphysical results")
    print("  - Input file formatting errors")
    print("\nTry running 'vplanet vpl.in' in the input directory to")
    print("diagnose the problem.")


if __name__ == '__main__':
    main()
