"""Optimizer wrapper for MaxLEV."""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Callable, Tuple, Dict, Any, List, Optional


class Optimizer:
    """Wrapper for scipy optimizers with consistent interface."""

    def __init__(self, config: Dict[str, Any]):
        self.algorithm = config.get('algorithm', 'differential_evolution')
        self.config = config

    def optimize(self, objective: Callable, bounds: np.ndarray,
                 param_names: list) -> Tuple[np.ndarray, float, dict]:
        """Run optimization and return best parameters, value, and info."""
        print(f"\nStarting maximum likelihood optimization...")
        print(f"Method: {self.algorithm}")
        print(f"Parameter space: {len(param_names)} dimensions")

        print("\nParameter bounds:")
        for i, name in enumerate(param_names):
            print(f"  {name:30s}: [{bounds[i,0]:12.6f}, {bounds[i,1]:12.6f}]")

        if self.algorithm == 'differential_evolution':
            result = self._fnRunDifferentialEvolution(objective, bounds)
        elif self.algorithm in ['powell', 'nelder-mead', 'bfgs', 'cobyla']:
            result = self._fnRunLocalOptimizer(objective, bounds)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        if result.success:
            print("\n[OK] Optimization converged successfully")
        else:
            print(f"\n[WARNING] Optimization did not fully converge: {result.message}")

        print("\nFinal parameters:")
        for i, name in enumerate(param_names):
            val = result.x[i]
            bInBounds = bounds[i,0] <= val <= bounds[i,1]
            sStatus = "[OK]" if bInBounds else "[OUT OF BOUNDS]"
            print(f"  {name:30s} = {val:12.6e} {sStatus}")

        return result.x, result.fun, {'success': result.success, 'message': result.message}

    def _fdaParseInitialGuess(self, bounds: np.ndarray) -> np.ndarray:
        """Return initial guess from config x0 or center of bounds."""
        daConfigX0 = self.config.get('x0')
        if daConfigX0 is not None:
            return np.array(daConfigX0, dtype=float)
        return np.mean(bounds, axis=1)

    def _fnRunDifferentialEvolution(self, objective: Callable,
                                    bounds: np.ndarray):
        """Run differential evolution optimizer."""
        dictDeConfig = self.config.get('de_settings', {})

        result = differential_evolution(
            objective,
            bounds,
            strategy=dictDeConfig.get('strategy', 'best1bin'),
            maxiter=self.config.get('maxiter', 1000),
            popsize=dictDeConfig.get('popsize', 15),
            tol=self.config.get('tol', 0.01),
            mutation=tuple(dictDeConfig.get('mutation', [0.5, 1.0])),
            recombination=dictDeConfig.get('recombination', 0.7),
            seed=self.config.get('seed', 42),
            workers=dictDeConfig.get('workers', 1),
            updating=dictDeConfig.get('updating', 'deferred'),
            polish=dictDeConfig.get('polish', False),
            disp=dictDeConfig.get('disp', True),
        )

        return result

    def _fnRunLocalOptimizer(self, objective: Callable,
                             bounds: np.ndarray):
        """Run local optimizer (Powell, Nelder-Mead, etc.)."""
        daX0 = self._fdaParseInitialGuess(bounds)
        sMethod = self.algorithm.lower()

        dictOptions = {
            'maxiter': self.config.get('maxiter', 1000),
            'disp': self.config.get('disp', True),
        }

        if sMethod == 'nelder-mead':
            dictOptions.update(
                self._fdictBuildNelderMeadOptions()
            )

        listScipyBounds = [(b[0], b[1]) for b in bounds]

        result = minimize(
            objective,
            daX0,
            method=sMethod,
            bounds=listScipyBounds if sMethod != 'nelder-mead' else None,
            options=dictOptions,
        )

        return result

    def _fdictBuildNelderMeadOptions(self) -> Dict[str, Any]:
        """Extract Nelder-Mead-specific options from config."""
        dictNmConfig = self.config.get('nm_settings', {})
        dictOptions = {}

        if 'adaptive' in dictNmConfig:
            dictOptions['adaptive'] = dictNmConfig['adaptive']
        if 'xatol' in dictNmConfig:
            dictOptions['xatol'] = dictNmConfig['xatol']
        if 'fatol' in dictNmConfig:
            dictOptions['fatol'] = dictNmConfig['fatol']
        if 'initial_simplex' in dictNmConfig:
            dictOptions['initial_simplex'] = np.array(
                dictNmConfig['initial_simplex']
            )

        return dictOptions
