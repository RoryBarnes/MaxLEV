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
