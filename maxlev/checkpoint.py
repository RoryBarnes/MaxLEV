"""
Checkpoint and resume support for differential evolution.

After every generation, MaxLEV writes the population and its objective
values to a JSON checkpoint. A run that was interrupted can then resume
from that population instead of starting over: the population becomes
the initial population of a new differential-evolution run, and the
stored objective values are returned without re-running VPLanet.
"""

import datetime
import json
import os

import numpy as np


CHECKPOINT_FORMAT = "maxlev-differential-evolution-checkpoint"
CHECKPOINT_VERSION = 1
MINIMUM_POPULATION_SIZE = 5
BOUNDS_RELATIVE_TOLERANCE = 1e-9


class CheckpointError(ValueError):
    """Raised when a checkpoint is missing, malformed, or incompatible."""
    pass


def fdictBuildCheckpoint(listParamNames, daBounds, daPopulation,
                         daPopulationEnergies, iGenerationsCompleted):
    """Return the JSON-serializable checkpoint of one population."""
    daEnergies = np.asarray(daPopulationEnergies, dtype=float)
    daPopulation = np.asarray(daPopulation, dtype=float)
    dictBest = fdictFindBestMember(daPopulation, daEnergies)
    return {
        "sFormat": CHECKPOINT_FORMAT,
        "iVersion": CHECKPOINT_VERSION,
        "sTimestamp": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "iGenerationsCompleted": int(iGenerationsCompleted),
        "listParameterNames": list(listParamNames),
        "daBounds": np.asarray(daBounds, dtype=float).tolist(),
        "daPopulation": daPopulation.tolist(),
        "daPopulationEnergies": flistEnergiesToJson(daEnergies),
        **dictBest,
    }


def fdictFindBestMember(daPopulation, daEnergies):
    """Return the lowest-energy member, or nulls if no energy is known."""
    if not np.any(np.isfinite(daEnergies)):
        return {"daBestParameters": None, "dBestValue": None}
    iBest = int(np.nanargmin(np.where(np.isfinite(daEnergies),
                                      daEnergies, np.nan)))
    return {"daBestParameters": daPopulation[iBest].tolist(),
            "dBestValue": float(daEnergies[iBest])}


def flistEnergiesToJson(daEnergies):
    """Return energies as a list with non-finite values written as null."""
    return [float(d) if np.isfinite(d) else None for d in daEnergies]


def fnWriteCheckpoint(sPath, dictCheckpoint):
    """Write a checkpoint atomically so a crash never leaves half a file."""
    sTemporaryPath = sPath + ".tmp"
    try:
        with open(sTemporaryPath, "w") as fileCheckpoint:
            json.dump(dictCheckpoint, fileCheckpoint, indent=1,
                      allow_nan=False)
        os.replace(sTemporaryPath, sPath)
    except (OSError, ValueError) as error:
        raise CheckpointError(
            f"Cannot write checkpoint {sPath}: {error}") from error


def fdictReadCheckpoint(sPath):
    """Return the checkpoint stored at sPath after checking its format."""
    try:
        with open(sPath) as fileCheckpoint:
            dictCheckpoint = json.load(fileCheckpoint)
    except FileNotFoundError:
        raise CheckpointError(f"Checkpoint file not found: {sPath}")
    except (OSError, ValueError) as error:
        raise CheckpointError(f"Cannot read checkpoint {sPath}: {error}")
    if not isinstance(dictCheckpoint, dict) or \
            dictCheckpoint.get("sFormat") != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"{sPath} is not a MaxLEV differential-evolution checkpoint")
    if dictCheckpoint.get("iVersion") != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"{sPath} has checkpoint version {dictCheckpoint.get('iVersion')}"
            f"; this MaxLEV reads version {CHECKPOINT_VERSION}")
    return dictCheckpoint


def fdaPopulationFromCheckpoint(dictCheckpoint):
    """Return the population array of a checkpoint."""
    try:
        daPopulation = np.array(dictCheckpoint["daPopulation"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(f"Checkpoint population is unreadable: {error}")
    if daPopulation.ndim != 2:
        raise CheckpointError(
            "Checkpoint population must be a list of parameter vectors")
    return daPopulation


def fdaEnergiesFromCheckpoint(dictCheckpoint, iPopulationSize):
    """Return the population energies, NaN where a value is unknown."""
    listEnergies = dictCheckpoint.get("daPopulationEnergies")
    if listEnergies is None:
        return np.full(iPopulationSize, np.nan)
    if len(listEnergies) != iPopulationSize:
        raise CheckpointError(
            f"Checkpoint has {len(listEnergies)} energies for "
            f"{iPopulationSize} population members")
    return np.array([np.nan if d is None else d for d in listEnergies],
                    dtype=float)


def fiGenerationsFromCheckpoint(dictCheckpoint):
    """Return the number of generations the checkpointed run completed."""
    iCompleted = dictCheckpoint.get("iGenerationsCompleted")
    if isinstance(iCompleted, bool) or not isinstance(iCompleted, int) \
            or iCompleted < 0:
        raise CheckpointError(
            "Checkpoint iGenerationsCompleted must be a non-negative "
            f"integer, not {iCompleted!r}")
    return iCompleted


def fnValidateCheckpoint(dictCheckpoint, listParamNames, daBounds):
    """Raise CheckpointError unless the checkpoint fits this problem."""
    listStoredNames = dictCheckpoint.get("listParameterNames")
    if listStoredNames != list(listParamNames):
        raise CheckpointError(
            f"Checkpoint parameters {listStoredNames} differ from the "
            f"configuration's {list(listParamNames)}")
    daStoredBounds = np.array(dictCheckpoint.get("daBounds"), dtype=float)
    if daStoredBounds.shape != np.shape(daBounds) or not np.allclose(
            daStoredBounds, daBounds, rtol=BOUNDS_RELATIVE_TOLERANCE,
            atol=0.0):
        raise CheckpointError(
            "Checkpoint parameter bounds differ from the configuration's; "
            "a resumed search must use the same parameter space")
    fnValidatePopulation(fdaPopulationFromCheckpoint(dictCheckpoint),
                         daBounds)


def fnValidatePopulation(daPopulation, daBounds):
    """Raise CheckpointError unless the population fits the bounds."""
    daBounds = np.asarray(daBounds, dtype=float)
    if daPopulation.shape[1] != daBounds.shape[0]:
        raise CheckpointError(
            f"Checkpoint population has {daPopulation.shape[1]} parameters; "
            f"the configuration has {daBounds.shape[0]}")
    if daPopulation.shape[0] < MINIMUM_POPULATION_SIZE:
        raise CheckpointError(
            f"Checkpoint population has {daPopulation.shape[0]} members; "
            f"differential evolution needs at least {MINIMUM_POPULATION_SIZE}")
    daSlack = BOUNDS_RELATIVE_TOLERANCE * (daBounds[:, 1] - daBounds[:, 0])
    if np.any(daPopulation < daBounds[:, 0] - daSlack) or \
            np.any(daPopulation > daBounds[:, 1] + daSlack):
        raise CheckpointError(
            "Checkpoint population lies outside the parameter bounds")


class CheckpointWriter:
    """Differential-evolution callback that checkpoints every generation."""

    def __init__(self, sPath, listParamNames, daBounds,
                 iGenerationOffset=0):
        self.sPath = sPath
        self.listParamNames = list(listParamNames)
        self.daBounds = np.asarray(daBounds, dtype=float)
        self.iGenerationOffset = int(iGenerationOffset)

    def __call__(self, intermediate_result):
        """Write the current population; never stop the optimizer."""
        dictCheckpoint = fdictBuildCheckpoint(
            self.listParamNames, self.daBounds,
            intermediate_result.population,
            intermediate_result.population_energies,
            self.iGenerationOffset + intermediate_result.nit,
        )
        fnWriteCheckpoint(self.sPath, dictCheckpoint)
        return False


class CachedObjective:
    """
    Objective that returns stored values for already-evaluated points.

    Resuming re-submits the checkpointed population to the optimizer,
    which evaluates it before the first new generation. Matching those
    points to their stored energies avoids repeating the simulations.
    Points are matched within a tiny fraction of each parameter's range,
    because the optimizer rescales the population internally.
    """

    def __init__(self, fnObjective, daKnownPoints, daKnownValues, daBounds):
        daKnownValues = np.asarray(daKnownValues, dtype=float)
        baKnown = np.isfinite(daKnownValues)
        self.fnObjective = fnObjective
        self.daKnownPoints = np.asarray(daKnownPoints, dtype=float)[baKnown]
        self.daKnownValues = daKnownValues[baKnown]
        daBounds = np.asarray(daBounds, dtype=float)
        self.daTolerance = BOUNDS_RELATIVE_TOLERANCE * (
            daBounds[:, 1] - daBounds[:, 0])

    def __call__(self, daTheta):
        """Return the stored value of daTheta, or evaluate it."""
        iMatch = self.fiFindKnownPoint(daTheta)
        if iMatch >= 0:
            return float(self.daKnownValues[iMatch])
        return self.fnObjective(daTheta)

    def fiFindKnownPoint(self, daTheta):
        """Return the index of the stored point equal to daTheta, or -1."""
        if len(self.daKnownPoints) == 0:
            return -1
        daDistance = np.abs(self.daKnownPoints - np.asarray(daTheta))
        baMatches = np.all(daDistance <= self.daTolerance, axis=1)
        iaMatches = np.flatnonzero(baMatches)
        return int(iaMatches[0]) if len(iaMatches) > 0 else -1
