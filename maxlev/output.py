"""Output generation for MaxLEV."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any


def save_results(best_params: np.ndarray, best_value: float,
                 config, model, output_settings: Dict[str, Any]):
    """Save optimization results to file."""

    filepath = output_settings.get('results_file', 'maxlike_results.txt')

    with open(filepath, 'w') as f:
        f.write(f"{config.name} Maximum Likelihood Estimation\n")
        f.write("=" * 70 + "\n\n")

        f.write("Maximum Likelihood Parameters:\n")
        f.write("-" * 70 + "\n")
        for i, param in enumerate(config.parameters):
            f.write(f"{param.name:30s} = {best_params[i]:.6e}\n")

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
