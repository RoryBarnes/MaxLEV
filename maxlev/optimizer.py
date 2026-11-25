"""Optimizer wrapper for MaxLEV."""

import os

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Callable, Tuple, Dict, Any, List, Optional

from .checkpoint import (
    CachedObjective,
    CheckpointError,
    CheckpointWriter,
    fdaEnergiesFromCheckpoint,
    fdaPopulationFromCheckpoint,
    fdictReadCheckpoint,
    fiGenerationsFromCheckpoint,
    fnValidateCheckpoint,
)


def fnPrintResumeSummary(sResumePath, iCompleted, iMaxIterations,
                         daEnergies):
    """Print where a resumed differential-evolution run starts."""
    iKnown = int(np.sum(np.isfinite(daEnergies)))
    print(f"\nResuming from checkpoint: {sResumePath}")
    print(f"  Generations completed: {iCompleted} of {iMaxIterations}")
    print(f"  Generations remaining: {max(iMaxIterations - iCompleted, 0)}")
    print(f"  Population: {len(daEnergies)} members, {iKnown} with stored "
          f"objective values, {len(daEnergies) - iKnown} to evaluate")


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
            result = self._fnRunDifferentialEvolution(
                objective, bounds, param_names)
        elif self.config.get('resume_file'):
            raise CheckpointError(
                "Only differential_evolution can resume from a checkpoint; "
                f"the configured algorithm is {self.algorithm}")
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
                                    bounds: np.ndarray, param_names: list):
        """Run differential evolution, resuming and checkpointing as set."""
        dictStart = self._fdictBuildStartingState(
            objective, bounds, param_names)
        checkpointWriter = self._fBuildCheckpointWriter(
            bounds, param_names, dictStart['iGenerationOffset'])
        dictKeywords = self._fdictBuildDifferentialEvolutionKeywords()
        dictKeywords.update(maxiter=dictStart['iMaxIterations'],
                            init=dictStart['init'], callback=checkpointWriter)
        result = differential_evolution(
            dictStart['objective'], bounds, **dictKeywords)
        if checkpointWriter is not None:
            checkpointWriter(intermediate_result=result)
        return result

    def _fdictBuildDifferentialEvolutionKeywords(self) -> Dict[str, Any]:
        """Return scipy differential_evolution settings from the config."""
        dictDeConfig = self.config.get('de_settings', {})
        return {
            'strategy': dictDeConfig.get('strategy', 'best1bin'),
            'popsize': dictDeConfig.get('popsize', 15),
            'tol': self.config.get('tol', 0.01),
            'mutation': tuple(dictDeConfig.get('mutation', [0.5, 1.0])),
            'recombination': dictDeConfig.get('recombination', 0.7),
            'seed': self.config.get('seed', 42),
            'workers': dictDeConfig.get('workers', 1),
            'updating': dictDeConfig.get('updating', 'deferred'),
            'polish': dictDeConfig.get('polish', False),
            'disp': dictDeConfig.get('disp', True),
        }

    def _fdictBuildStartingState(self, objective: Callable,
                                 bounds: np.ndarray,
                                 param_names: list) -> Dict[str, Any]:
        """Return objective, initial population and generation budget."""
        iMaxIterations = self.config.get('maxiter', 1000)
        sResumePath = self.config.get('resume_file')
        if not sResumePath:
            self._fnRefuseToOverwriteCheckpoint()
            return {'objective': objective, 'init': 'latinhypercube',
                    'iGenerationOffset': 0, 'iMaxIterations': iMaxIterations}
        dictCheckpoint = fdictReadCheckpoint(sResumePath)
        fnValidateCheckpoint(dictCheckpoint, param_names, bounds)
        daPopulation = fdaPopulationFromCheckpoint(dictCheckpoint)
        daEnergies = fdaEnergiesFromCheckpoint(
            dictCheckpoint, len(daPopulation))
        iCompleted = fiGenerationsFromCheckpoint(dictCheckpoint)
        fnPrintResumeSummary(sResumePath, iCompleted, iMaxIterations,
                             daEnergies)
        return {'objective': CachedObjective(
                    objective, daPopulation, daEnergies, bounds),
                'init': daPopulation, 'iGenerationOffset': iCompleted,
                'iMaxIterations': max(iMaxIterations - iCompleted, 0)}

    def _fnRefuseToOverwriteCheckpoint(self) -> None:
        """Stop a fresh run that would overwrite an earlier checkpoint."""
        sCheckpointPath = self.config.get('checkpoint_file')
        if sCheckpointPath and os.path.exists(sCheckpointPath):
            raise CheckpointError(
                f"Checkpoint {sCheckpointPath} exists from an earlier run. "
                "Continue that run with --resume, or move or delete the "
                "file to start a new search.")

    def _fBuildCheckpointWriter(self, bounds: np.ndarray, param_names: list,
                                iGenerationOffset: int):
        """Return the per-generation checkpoint callback, or None."""
        sCheckpointPath = self.config.get('checkpoint_file')
        if not sCheckpointPath:
            return None
        print(f"Checkpoint after every generation: {sCheckpointPath}")
        return CheckpointWriter(sCheckpointPath, param_names, bounds,
                                iGenerationOffset)

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
