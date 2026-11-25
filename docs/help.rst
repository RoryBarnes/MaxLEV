Running MaxLEV
==============

Run MaxLEV with the command:

.. code-block:: bash

    maxlev <config.json>

where ``config.json`` is a JSON file that defines the optimization problem (see
:doc:`config` for details). The ``maxlev`` command is available after installation
with ``pip install -e .`` (see :doc:`install`).

Command-Line Options
--------------------

``--validate``
    Validate configuration without running optimization. Useful for checking that
    all parameters and outputs are valid before a long optimization run.

``--verbose, -v``
    Enable verbose VPLanet output. By default, VPLanet output is suppressed during
    optimization.

``--seed N``
    Override the random seed specified in the configuration file.

``--maxiter N``
    Override the maximum number of iterations.

``--workers N``
    Override the number of parallel workers:

    - ``1``: Serial execution (default)
    - ``-1``: Use all available CPUs
    - ``N``: Use N parallel workers

``--resume [CHECKPOINT]``
    Continue an interrupted differential-evolution run from a checkpoint
    instead of starting over. Without a path, ``MaxLEV`` reads the configured
    checkpoint file (``<name>_checkpoint.json`` by default). See
    :ref:`resuming`.

Examples
--------

Validate a configuration file:

.. code-block:: bash

    maxlev myconfig.json --validate

Run optimization with 4 parallel workers:

.. code-block:: bash

    maxlev myconfig.json --workers 4

Run with a different random seed:

.. code-block:: bash

    maxlev myconfig.json --seed 12345

Continue a run that stopped (power loss, killed job, closed session):

.. code-block:: bash

    maxlev myconfig.json --resume

Output Files
------------

After a successful run, ``MaxLEV`` generates:

1. **Results file** (default: ``maxlike_results.txt``): Contains the best-fit
   parameters, negative log-likelihood, chi-squared, model predictions with
   residuals, and parameter bounds.

2. **MaxLEV input files** (``*_maxlev.in``): For each VPLanet input file that
   contains optimized parameters, a new file is created with the maximum
   likelihood values substituted. These files are placed in the same directory
   as the template files.

3. **Checkpoint file** (differential evolution only; default:
   ``<name>_checkpoint.json``): The population and its objective values,
   rewritten after every generation and once more when the run ends. Use it
   with ``--resume`` to continue or extend the search (see :ref:`resuming`).

Example output structure:

.. code-block:: text

    examples/<Method>/
        earth.in           # Original template
        earth_maxlev.in    # Generated with ML values
        sun.in
        vpl.in
    <method>_results.txt

Timeout Handling
----------------

VPLanet simulations can sometimes hang or take extremely long for certain
parameter combinations. ``MaxLEV`` includes timeout protection:

.. code-block:: json

    {
        "vplanet": {
            "timeout": 120
        }
    }

If a simulation exceeds the timeout (in seconds), it is terminated and the
parameter combination is assigned the failure penalty value.

.. _resuming:

Resuming an Interrupted Run
---------------------------

A differential-evolution search with an expensive forward model can run for
hours or days, and losing it to a crash or a closed session means losing every
simulation it has done. ``MaxLEV`` therefore writes a checkpoint after every
generation: a JSON file holding the population (one parameter vector per
member), each member's objective value, the number of generations completed,
and the parameter names and bounds. The file is replaced atomically, so an
interruption never leaves a partial checkpoint.

To continue, rerun the same configuration with ``--resume``:

.. code-block:: bash

    maxlev myconfig.json --resume

The checkpointed population becomes the initial population of a new
differential-evolution run (Storn & Price 1997, *J. Global Optim.* 11, 341),
and the run stops after ``maxiter`` generations in total, counting those
completed before the interruption. The stored objective values are reused, so
no simulation is repeated. To extend a finished search, resume it with a
larger ``--maxiter``.

Rules that protect the result:

- ``MaxLEV`` refuses to resume when the checkpoint's parameter names or bounds
  differ from the configuration's, because the population would then belong
  to a different problem.
- A new run (without ``--resume``) refuses to start while a checkpoint from an
  earlier run exists at the checkpoint path, so a search is never overwritten
  by accident. Resume it, or move or delete the file to start over.
- A resumed run is reproducible: resuming the same checkpoint with the same
  seed gives the same result. It does not retrace the path the uninterrupted
  run would have taken, because the random-number generator's state is not
  stored. Either way, the best member never gets worse when a run resumes.
- Only ``differential_evolution`` writes and reads checkpoints.

A checkpoint may also omit ``daPopulationEnergies`` (or contain ``null`` for
individual members), for example when a population is rebuilt from an
external evaluation log. Those members are evaluated when the run resumes, and
resuming with ``--maxiter`` equal to the completed generations evaluates the
population without running a new generation and saves the values in a fresh
checkpoint.
