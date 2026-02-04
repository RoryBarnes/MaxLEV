"""Command-line interface for MaxLEV."""

import argparse
import sys
from pathlib import Path


def parse_args():
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

    parser.add_argument('config', type=str,
                       help='Path to configuration file (JSON)')

    parser.add_argument('--validate', action='store_true',
                       help='Validate configuration without running optimization')

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

    return parser.parse_args()


def main():
    """Main entry point for MaxLEV CLI."""
    args = parse_args()

    # Import here to allow --help without full imports
    from .config import MaxLEVConfig
    from .likelihood import create_likelihood
    from .observables import ObservableComputer
    from .model import MaxLEVModel
    from .optimizer import Optimizer
    from .output import save_results, plot_evolution

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    print("=" * 70)
    print("VPlanet Maximum Likelihood Estimation")
    print("=" * 70)
    print(f"\nLoading configuration: {config_path}")

    if not config_path.suffix == '.json':
        print(f"Error: Configuration file must be JSON format")
        sys.exit(1)

    config = MaxLEVConfig.from_json(str(config_path))

    print(f"  Name: {config.name}")
    print(f"  Parameters: {len(config.parameters)}")
    print(f"  Observables: {len(config.observables)}")

    # Apply command-line overrides
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

    # Validate configuration
    print("\nValidating configuration...")
    errors = config.validate()
    if errors:
        print("\n" + "=" * 70)
        print("CONFIGURATION ERRORS")
        print("=" * 70)
        for error_msg, line_num in errors:
            if line_num > 0:
                print(f"Line {line_num}: {error_msg}")
            else:
                print(f"{error_msg}")
        print("\nPlease fix the errors above and try again.")
        sys.exit(1)

    print("[OK] Configuration validated")

    if args.validate:
        print("\nValidation complete (--validate flag, not running optimization)")
        sys.exit(0)

    # Initialize components
    print("\nInitializing VPlanet model...")
    output_names = [o.name for o in config.outputs]

    likelihood_model = create_likelihood(config.likelihood, config.observables)
    observable_computer = ObservableComputer(output_names, config.observables)
    model = MaxLEVModel(config, likelihood_model, observable_computer)
    print("[OK] VPlanet model initialized")

    # Run optimization
    optimizer = Optimizer(config.optimizer)
    best_params, best_value, result_info = optimizer.optimize(
        model.neg_log_likelihood,
        model.bounds,
        model.param_names
    )

    # Report results
    print("\n" + "=" * 70)
    print("Maximum Likelihood Results")
    print("=" * 70)
    for i, param in enumerate(config.parameters):
        print(f"{param.name:30s} = {best_params[i]:.6e}")
    print(f"\n-ln(Likelihood) = {best_value:.6e}")
    print(f"chi^2           = {2*best_value:.6e}")

    # Save results and generate plots
    save_results(best_params, best_value, config, model, config.output_settings)
    plot_evolution(best_params, config, model, config.output_settings)

    print("\n" + "=" * 70)
    print("Maximum likelihood estimation completed successfully!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
