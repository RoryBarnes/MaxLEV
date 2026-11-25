"""Test checkpointing and resuming differential evolution in MaxLEV."""

import json
import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maxlev.checkpoint import (
    CHECKPOINT_FORMAT,
    CachedObjective,
    CheckpointError,
    CheckpointWriter,
    fdaEnergiesFromCheckpoint,
    fdaPopulationFromCheckpoint,
    fdictBuildCheckpoint,
    fdictReadCheckpoint,
    fiGenerationsFromCheckpoint,
    fnValidateCheckpoint,
    fnWriteCheckpoint,
)
from maxlev.optimizer import Optimizer

LIST_PARAMETER_NAMES = ['x1', 'x2']
DA_BOUNDS = np.array([[-5.0, 5.0], [0.0, 10.0]])
I_POPULATION_SIZE = 10


def fdQuadratic(daTheta):
    """Return a smooth objective with its minimum at (1, 3)."""
    return float((daTheta[0] - 1.0) ** 2 + (daTheta[1] - 3.0) ** 2)


class CountingObjective:
    """Picklable objective that counts how often it is evaluated."""

    def __init__(self):
        self.iCalls = 0

    def __call__(self, daTheta):
        self.iCalls += 1
        return fdQuadratic(daTheta)


def fdaBuildPopulation():
    """Return a fixed population inside DA_BOUNDS."""
    daUnit = np.random.default_rng(7).uniform(size=(I_POPULATION_SIZE, 2))
    return DA_BOUNDS[:, 0] + daUnit * (DA_BOUNDS[:, 1] - DA_BOUNDS[:, 0])


def fdictBuildSampleCheckpoint(iGenerationsCompleted=3):
    """Return a checkpoint of the fixed population with its energies."""
    daPopulation = fdaBuildPopulation()
    daEnergies = np.array([fdQuadratic(d) for d in daPopulation])
    return fdictBuildCheckpoint(LIST_PARAMETER_NAMES, DA_BOUNDS, daPopulation,
                                daEnergies, iGenerationsCompleted)


def fdictBuildOptimizerConfig(sCheckpointPath, iMaxIterations,
                              sResumePath=None):
    """Return a small serial differential-evolution configuration."""
    dictConfig = {
        'algorithm': 'differential_evolution', 'maxiter': iMaxIterations,
        'seed': 11, 'tol': 0.0, 'checkpoint_file': sCheckpointPath,
        'de_settings': {'popsize': 5, 'workers': 1, 'disp': False},
    }
    if sResumePath is not None:
        dictConfig['resume_file'] = sResumePath
    return dictConfig


def fnRunOptimizer(dictConfig, fnObjective):
    """Run the optimizer on the test problem and return its result."""
    return Optimizer(dictConfig).optimize(
        fnObjective, DA_BOUNDS, LIST_PARAMETER_NAMES)


class TestCheckpointFile:
    """Tests for writing and reading checkpoint files."""

    def test_round_trip_preserves_population(self, tmp_path):
        """Test that a written checkpoint reads back unchanged."""
        sPath = str(tmp_path / 'run_checkpoint.json')
        dictCheckpoint = fdictBuildSampleCheckpoint()
        fnWriteCheckpoint(sPath, dictCheckpoint)
        dictRead = fdictReadCheckpoint(sPath)
        np.testing.assert_array_equal(fdaPopulationFromCheckpoint(dictRead),
                                      fdaBuildPopulation())
        assert dictRead['iGenerationsCompleted'] == 3

    def test_write_leaves_no_temporary_file(self, tmp_path):
        """Test that the atomic write removes its temporary file."""
        sPath = str(tmp_path / 'run_checkpoint.json')
        fnWriteCheckpoint(sPath, fdictBuildSampleCheckpoint())
        assert [p.name for p in tmp_path.iterdir()] == ['run_checkpoint.json']

    def test_best_member_recorded(self):
        """Test that the best parameters and value are stored."""
        dictCheckpoint = fdictBuildSampleCheckpoint()
        daEnergies = np.array(dictCheckpoint['daPopulationEnergies'])
        assert dictCheckpoint['dBestValue'] == pytest.approx(daEnergies.min())
        assert dictCheckpoint['daBestParameters'] == \
            dictCheckpoint['daPopulation'][int(np.argmin(daEnergies))]

    def test_nonfinite_energy_written_as_null(self):
        """Test that unknown energies become JSON null."""
        dictCheckpoint = fdictBuildCheckpoint(
            LIST_PARAMETER_NAMES, DA_BOUNDS, fdaBuildPopulation()[:5],
            [1.0, np.nan, 2.0, np.inf, 3.0], 0)
        assert dictCheckpoint['daPopulationEnergies'][1] is None
        assert dictCheckpoint['daPopulationEnergies'][3] is None
        assert dictCheckpoint['dBestValue'] == 1.0

    def test_unknown_energies_leave_best_null(self, tmp_path):
        """Test that a population with no energies is still valid JSON."""
        sPath = str(tmp_path / 'population.json')
        dictCheckpoint = fdictBuildCheckpoint(
            LIST_PARAMETER_NAMES, DA_BOUNDS, fdaBuildPopulation(),
            np.full(I_POPULATION_SIZE, np.nan), 0)
        fnWriteCheckpoint(sPath, dictCheckpoint)
        dictRead = json.loads(Path(sPath).read_text())
        assert dictRead['dBestValue'] is None
        assert dictRead['daBestParameters'] is None

    def test_missing_file_raises(self, tmp_path):
        """Test a clear error for a missing checkpoint."""
        with pytest.raises(CheckpointError, match='not found'):
            fdictReadCheckpoint(str(tmp_path / 'absent.json'))

    def test_invalid_json_raises(self, tmp_path):
        """Test a clear error for an unreadable checkpoint."""
        pathBad = tmp_path / 'bad.json'
        pathBad.write_text('{not json')
        with pytest.raises(CheckpointError, match='Cannot read'):
            fdictReadCheckpoint(str(pathBad))

    def test_foreign_json_raises(self, tmp_path):
        """Test that a JSON file of another kind is rejected."""
        pathOther = tmp_path / 'other.json'
        pathOther.write_text(json.dumps({'name': 'config'}))
        with pytest.raises(CheckpointError, match='not a MaxLEV'):
            fdictReadCheckpoint(str(pathOther))

    def test_wrong_version_raises(self, tmp_path):
        """Test that an unknown checkpoint version is rejected."""
        pathOld = tmp_path / 'old.json'
        pathOld.write_text(json.dumps(
            {'sFormat': CHECKPOINT_FORMAT, 'iVersion': 99}))
        with pytest.raises(CheckpointError, match='version 99'):
            fdictReadCheckpoint(str(pathOld))

    def test_unwritable_path_raises(self, tmp_path):
        """Test a clear error when the checkpoint cannot be written."""
        sPath = str(tmp_path / 'missing_directory' / 'run.json')
        with pytest.raises(CheckpointError, match='Cannot write'):
            fnWriteCheckpoint(sPath, fdictBuildSampleCheckpoint())


class TestCheckpointValidation:
    """Tests for matching a checkpoint to the configured problem."""

    def test_matching_checkpoint_passes(self):
        """Test that a checkpoint of the same problem validates."""
        fnValidateCheckpoint(fdictBuildSampleCheckpoint(),
                             LIST_PARAMETER_NAMES, DA_BOUNDS)

    def test_different_parameters_raise(self):
        """Test that renamed parameters are rejected."""
        with pytest.raises(CheckpointError, match='parameters'):
            fnValidateCheckpoint(fdictBuildSampleCheckpoint(),
                                 ['x1', 'y2'], DA_BOUNDS)

    def test_different_bounds_raise(self):
        """Test that a changed parameter space is rejected."""
        daWider = DA_BOUNDS.copy()
        daWider[1, 1] = 20.0
        with pytest.raises(CheckpointError, match='bounds'):
            fnValidateCheckpoint(fdictBuildSampleCheckpoint(),
                                 LIST_PARAMETER_NAMES, daWider)

    def test_population_outside_bounds_raises(self):
        """Test that a member outside the bounds is rejected."""
        dictCheckpoint = fdictBuildSampleCheckpoint()
        dictCheckpoint['daPopulation'][0][1] = 11.0
        with pytest.raises(CheckpointError, match='outside'):
            fnValidateCheckpoint(dictCheckpoint, LIST_PARAMETER_NAMES,
                                 DA_BOUNDS)

    def test_small_population_raises(self):
        """Test that fewer than five members are rejected."""
        dictCheckpoint = fdictBuildSampleCheckpoint()
        dictCheckpoint['daPopulation'] = dictCheckpoint['daPopulation'][:4]
        with pytest.raises(CheckpointError, match='at least 5'):
            fnValidateCheckpoint(dictCheckpoint, LIST_PARAMETER_NAMES,
                                 DA_BOUNDS)

    def test_ragged_population_raises(self):
        """Test that a population that is not a matrix is rejected."""
        with pytest.raises(CheckpointError, match='population'):
            fdaPopulationFromCheckpoint({'daPopulation': [[1.0, 2.0], [3.0]]})

    def test_missing_energies_are_unknown(self):
        """Test that a checkpoint without energies yields NaN."""
        daEnergies = fdaEnergiesFromCheckpoint({}, 6)
        assert daEnergies.shape == (6,) and np.all(np.isnan(daEnergies))

    def test_energy_count_mismatch_raises(self):
        """Test that energies must match the population size."""
        with pytest.raises(CheckpointError, match='energies'):
            fdaEnergiesFromCheckpoint({'daPopulationEnergies': [1.0]}, 6)

    def test_negative_generations_raise(self):
        """Test that a negative generation count is rejected."""
        with pytest.raises(CheckpointError, match='non-negative'):
            fiGenerationsFromCheckpoint({'iGenerationsCompleted': -1})

    def test_missing_generations_raise(self):
        """Test that a checkpoint must state its generation count."""
        with pytest.raises(CheckpointError, match='non-negative'):
            fiGenerationsFromCheckpoint({})


class TestCachedObjective:
    """Tests for returning stored energies of known points."""

    def test_known_point_skips_evaluation(self):
        """Test that a stored point is not evaluated again."""
        fnCounter = CountingObjective()
        daPopulation = fdaBuildPopulation()
        cachedObjective = CachedObjective(
            fnCounter, daPopulation, np.arange(10.0), DA_BOUNDS)
        assert cachedObjective(daPopulation[4]) == 4.0
        assert fnCounter.iCalls == 0

    def test_rescaled_point_still_matches(self):
        """Test matching despite round-off from the optimizer's rescaling."""
        daPopulation = fdaBuildPopulation()
        daRange = DA_BOUNDS[:, 1] - DA_BOUNDS[:, 0]
        daUnit = (daPopulation[2] - DA_BOUNDS[:, 0]) / daRange
        daRescaled = DA_BOUNDS[:, 0] + daUnit * daRange
        cachedObjective = CachedObjective(
            CountingObjective(), daPopulation, np.arange(10.0), DA_BOUNDS)
        assert cachedObjective(daRescaled) == 2.0

    def test_unknown_point_is_evaluated(self):
        """Test that a new point is passed to the objective."""
        fnCounter = CountingObjective()
        cachedObjective = CachedObjective(
            fnCounter, fdaBuildPopulation(), np.arange(10.0), DA_BOUNDS)
        assert cachedObjective(np.array([1.0, 3.0])) == 0.0
        assert fnCounter.iCalls == 1

    def test_unknown_energy_is_evaluated(self):
        """Test that a member with a NaN energy is evaluated."""
        fnCounter = CountingObjective()
        daPopulation = fdaBuildPopulation()
        cachedObjective = CachedObjective(
            fnCounter, daPopulation, np.full(10, np.nan), DA_BOUNDS)
        assert cachedObjective(daPopulation[0]) == fdQuadratic(daPopulation[0])
        assert fnCounter.iCalls == 1

    def test_pickles_for_parallel_workers(self):
        """Test that the cache survives pickling for worker processes."""
        cachedObjective = CachedObjective(
            fdQuadratic, fdaBuildPopulation(), np.arange(10.0), DA_BOUNDS)
        cachedCopy = pickle.loads(pickle.dumps(cachedObjective))
        assert cachedCopy(fdaBuildPopulation()[7]) == 7.0


class TestCheckpointWriter:
    """Tests for the per-generation callback."""

    def test_writer_offsets_generations(self, tmp_path):
        """Test that resumed runs count generations from the checkpoint."""
        sPath = str(tmp_path / 'run_checkpoint.json')
        writer = CheckpointWriter(sPath, LIST_PARAMETER_NAMES, DA_BOUNDS,
                                  iGenerationOffset=30)
        daPopulation = fdaBuildPopulation()

        class IntermediateResult:
            population = daPopulation
            population_energies = np.arange(10.0)
            nit = 2

        assert writer(intermediate_result=IntermediateResult()) is False
        assert fdictReadCheckpoint(sPath)['iGenerationsCompleted'] == 32


class TestOptimizerCheckpointing:
    """Tests for checkpointing and resuming inside the optimizer."""

    def test_checkpoint_written_each_generation(self, tmp_path):
        """Test that a run leaves a checkpoint of its last generation."""
        sPath = str(tmp_path / 'run_checkpoint.json')
        fnRunOptimizer(fdictBuildOptimizerConfig(sPath, 3), fdQuadratic)
        dictCheckpoint = fdictReadCheckpoint(sPath)
        assert dictCheckpoint['iGenerationsCompleted'] == 3
        assert np.shape(dictCheckpoint['daPopulation']) == (10, 2)

    def test_no_checkpoint_when_disabled(self, tmp_path):
        """Test that checkpoint_file None writes nothing."""
        fnRunOptimizer(fdictBuildOptimizerConfig(None, 2), fdQuadratic)
        assert list(tmp_path.iterdir()) == []

    def test_resume_skips_stored_population(self, tmp_path):
        """Test that resuming evaluates only the new generations."""
        sPath = str(tmp_path / 'run_checkpoint.json')
        fnRunOptimizer(fdictBuildOptimizerConfig(sPath, 3), fdQuadratic)
        fnCounter = CountingObjective()
        fnRunOptimizer(fdictBuildOptimizerConfig(sPath, 5, sPath), fnCounter)
        assert fnCounter.iCalls == 2 * 10
        assert fdictReadCheckpoint(sPath)['iGenerationsCompleted'] == 5

    def test_resume_never_worsens_best(self, tmp_path):
        """Test that the resumed best is at least the checkpointed best."""
        sPath = str(tmp_path / 'run_checkpoint.json')
        fnRunOptimizer(fdictBuildOptimizerConfig(sPath, 3), fdQuadratic)
        dCheckpointBest = fdictReadCheckpoint(sPath)['dBestValue']
        _, dResumedBest, _ = fnRunOptimizer(
            fdictBuildOptimizerConfig(sPath, 8, sPath), fdQuadratic)
        assert dResumedBest <= dCheckpointBest

    def test_resume_is_reproducible(self, tmp_path):
        """Test that resuming one checkpoint twice gives the same answer."""
        sPath = str(tmp_path / 'run_checkpoint.json')
        fnWriteCheckpoint(sPath, fdictBuildSampleCheckpoint())
        listResults = []
        for sName in ['first.json', 'second.json']:
            dictConfig = fdictBuildOptimizerConfig(
                str(tmp_path / sName), 6, sPath)
            listResults.append(fnRunOptimizer(dictConfig, fdQuadratic))
        np.testing.assert_array_equal(listResults[0][0], listResults[1][0])

    def test_resume_without_energies_evaluates_population(self, tmp_path):
        """Test that a population with unknown energies is evaluated."""
        sPath = str(tmp_path / 'population.json')
        dictCheckpoint = fdictBuildSampleCheckpoint(iGenerationsCompleted=0)
        dictCheckpoint['daPopulationEnergies'] = None
        fnWriteCheckpoint(sPath, dictCheckpoint)
        fnCounter = CountingObjective()
        fnRunOptimizer(fdictBuildOptimizerConfig(
            str(tmp_path / 'run.json'), 1, sPath), fnCounter)
        assert fnCounter.iCalls == 10 + 10

    def test_finished_checkpoint_runs_no_generations(self, tmp_path):
        """Test that a completed budget returns the stored best directly."""
        sPath = str(tmp_path / 'run_checkpoint.json')
        dictCheckpoint = fdictBuildSampleCheckpoint(iGenerationsCompleted=5)
        fnWriteCheckpoint(sPath, dictCheckpoint)
        fnCounter = CountingObjective()
        _, dBest, _ = fnRunOptimizer(fdictBuildOptimizerConfig(
            str(tmp_path / 'run.json'), 5, sPath), fnCounter)
        assert fnCounter.iCalls == 0
        assert dBest == pytest.approx(dictCheckpoint['dBestValue'])

    def test_zero_generation_resume_stores_energies(self, tmp_path):
        """Test that evaluating a population without new generations saves its energies."""
        sPopulationPath = str(tmp_path / 'population.json')
        dictCheckpoint = fdictBuildSampleCheckpoint(iGenerationsCompleted=4)
        dictCheckpoint['daPopulationEnergies'] = None
        fnWriteCheckpoint(sPopulationPath, dictCheckpoint)
        sCheckpointPath = str(tmp_path / 'run_checkpoint.json')
        fnRunOptimizer(fdictBuildOptimizerConfig(
            sCheckpointPath, 4, sPopulationPath), fdQuadratic)
        dictWritten = fdictReadCheckpoint(sCheckpointPath)
        assert dictWritten['iGenerationsCompleted'] == 4
        assert None not in dictWritten['daPopulationEnergies']

    def test_resume_with_parallel_workers(self, tmp_path):
        """Test checkpointing and resuming through a process pool."""
        sPath = str(tmp_path / 'run_checkpoint.json')
        dictConfig = fdictBuildOptimizerConfig(sPath, 2)
        dictConfig['de_settings']['workers'] = 2
        fnRunOptimizer(dictConfig, fdQuadratic)
        dictConfig.update(maxiter=4, resume_file=sPath)
        _, dBest, _ = fnRunOptimizer(dictConfig, fdQuadratic)
        dictCheckpoint = fdictReadCheckpoint(sPath)
        assert dictCheckpoint['iGenerationsCompleted'] == 4
        assert dBest == pytest.approx(dictCheckpoint['dBestValue'])

    def test_fresh_run_refuses_to_overwrite(self, tmp_path):
        """Test that an existing checkpoint is protected."""
        sPath = str(tmp_path / 'run_checkpoint.json')
        fnWriteCheckpoint(sPath, fdictBuildSampleCheckpoint())
        with pytest.raises(CheckpointError, match='--resume'):
            fnRunOptimizer(fdictBuildOptimizerConfig(sPath, 2), fdQuadratic)

    def test_local_optimizer_cannot_resume(self, tmp_path):
        """Test that only differential evolution accepts a checkpoint."""
        dictConfig = {'algorithm': 'powell', 'resume_file': 'x.json'}
        with pytest.raises(CheckpointError, match='Only differential'):
            fnRunOptimizer(dictConfig, fdQuadratic)
