Configuration Format
====================

``MaxLEV`` uses JSON configuration files to define optimization problems. This page
describes all available options.

Basic Structure
---------------

A complete configuration file has the following sections:

.. code-block:: json

    {
        "name": "MyOptimization",
        "vplanet": { ... },
        "parameters": [ ... ],
        "outputs": [ ... ],
        "observables": [ ... ],
        "likelihood": { ... },
        "optimizer": { ... },
        "output": { ... }
    }

VPLanet Settings
----------------

The ``vplanet`` section configures the VPLanet simulation:

.. code-block:: json

    {
        "vplanet": {
            "inpath": "examples/EarthInterior",
            "executable": "/path/to/vplanet",
            "vplfile": "vpl.in",
            "verbose": false,
            "timeout": 120
        }
    }

- ``inpath``: Directory containing the template VPLanet input files
- ``executable``: Path to the VPLanet executable (default: ``vplanet``)
- ``vplfile``: Name of the primary input file (default: ``vpl.in``)
- ``verbose``: Enable VPLanet verbose output (default: ``false``)
- ``timeout``: Maximum time in seconds for a single simulation (default: ``120``)

Parameters
----------

The ``parameters`` section defines which VPLanet input parameters to optimize:

.. code-block:: json

    {
        "parameters": [
            {
                "name": "earth.dTMan",
                "bounds": [2500, 3500],
                "units": "K",
                "description": "Initial mantle temperature"
            }
        ]
    }

- ``name``: Parameter name in ``filename.parameter`` format (e.g., ``earth.dTMan``
  refers to ``dTMan`` in ``earth.in``)
- ``bounds``: ``[min, max]`` bounds for the parameter
- ``units``: Physical units (used for display and validation)
- ``description``: Optional human-readable description

.. note::

    Parameter names are validated against VPLanet's ``-h`` output. Invalid
    parameter names will cause an error before optimization begins.

Shared Parameters
^^^^^^^^^^^^^^^^^

When multiple VPLanet body files share the same parameter value (e.g., six
planets with identical surface albedo), use the ``bodies`` list to declare a
shared parameter:

.. code-block:: json

    {
        "parameters": [
            {
                "name": "dIceAlbedo",
                "bodies": ["planet1", "planet2", "planet3"],
                "bounds": [0.4, 0.8],
                "units": "dimensionless",
                "description": "Ice albedo (shared across all planets)"
            }
        ]
    }

- ``name``: The VPLanet option name **without** a body prefix
- ``bodies``: List of body file names that receive this parameter value

The optimizer treats each shared parameter as a single free dimension. Internally
MaxLEV expands the parameter to every listed body when calling
``vplanet_inference``, ensuring that all bodies receive the same optimized value.

Rules:

- A shared parameter must list at least 2 bodies.
- The ``name`` must not use ``body.param`` format (no dots).
- Body names must not be duplicated within a single parameter.
- A shared parameter must not conflict with a non-shared parameter that targets
  the same ``body.param`` combination.

Priors
^^^^^^

Each parameter can optionally specify a prior distribution used for Maximum A
Posteriori (MAP) estimation:

.. code-block:: json

    {
        "name": "star.dMass",
        "bounds": [0.5, 1.5],
        "units": "Msun",
        "prior": {
            "type": "gaussian",
            "mean": 1.0,
            "std": 0.1
        }
    }

Supported prior types:

- ``uniform`` (default): Flat prior within bounds. Equivalent to no prior.
- ``gaussian``: Normal distribution with ``mean`` and ``std``.
- ``asymmetric_gaussian``: Asymmetric normal with ``mean``, ``std_upper``, and
  ``std_lower``.
- ``log_uniform``: Log-uniform (Jeffreys) prior, :math:`p(x) \propto 1/x`.
  Appropriate for scale parameters that span orders of magnitude (e.g.,
  viscosity, initial volatile inventory). Bounds must be strictly positive.

When any parameter has a non-uniform prior, MaxLEV automatically optimizes the
posterior (MAP) instead of the likelihood (MLE).

Outputs
-------

The ``outputs`` section defines which VPLanet output parameters to extract:

.. code-block:: json

    {
        "outputs": [
            {
                "name": "final.earth.TUMan",
                "units": "K"
            },
            {
                "name": "final.earth.HflowUMan",
                "units": "TW",
                "conversion_factor": 1e-12,
                "description": "Heat flow converted to TW"
            }
        ]
    }

- ``name``: Output parameter in ``final.body.parameter`` format
- ``units``: Physical units of the output
- ``conversion_factor``: Optional multiplier applied to raw VPLanet output (default: ``1.0``)
- ``description``: Optional description

.. warning::

    VPLanet outputs are sorted alphabetically by ``vplanet_inference``. The
    ``conversion_factor`` is applied after sorting, so ensure factors match
    the alphabetically sorted output order.

Observables
-----------

The ``observables`` section defines observational constraints:

.. code-block:: json

    {
        "observables": [
            {
                "name": "upperMantleTemperature",
                "type": "direct",
                "output": "final.earth.TUMan",
                "observed_value": 1587,
                "uncertainty": 100
            }
        ]
    }

Direct Observables
^^^^^^^^^^^^^^^^^^

Direct observables compare a VPLanet output directly to an observation:

.. code-block:: json

    {
        "name": "temperature",
        "type": "direct",
        "output": "final.earth.TUMan",
        "observed_value": 1587,
        "uncertainty": 100
    }

Asymmetric Uncertainties
^^^^^^^^^^^^^^^^^^^^^^^^

For observables with asymmetric error bars:

.. code-block:: json

    {
        "name": "temperature",
        "type": "direct",
        "output": "final.earth.TUMan",
        "observed_value": 1587,
        "uncertainty_lower": 34,
        "uncertainty_upper": 164
    }

The appropriate uncertainty is chosen based on whether the model value is above
or below the observed value.

Derived Observables
^^^^^^^^^^^^^^^^^^^

Derived observables compute a value from multiple outputs using a Python expression:

.. code-block:: json

    {
        "name": "luminosityRatio",
        "type": "derived",
        "expression": "outputs['final.star.Luminosity'] / outputs['final.star.LXUVStellar']",
        "observed_value": 1000,
        "uncertainty": 100
    }

Likelihood
----------

The ``likelihood`` section configures the likelihood function:

.. code-block:: json

    {
        "likelihood": {
            "type": "gaussian",
            "failure_penalty": 1e10
        }
    }

- ``type``: Likelihood type (currently only ``gaussian`` is supported)
- ``failure_penalty``: Value returned for failed simulations (default: ``1e10``)
- ``failure_check_window``: Number of consecutive failures that triggers an early
  abort (default: ``10``). If all simulations within the window fail, MaxLEV
  raises ``AllSimulationsFailedError`` to avoid wasting time on a misconfigured
  problem.

The Gaussian likelihood assumes independent observables:

.. math::

    -\ln L = \frac{1}{2} \sum_i \left(\frac{x_i - \mu_i}{\sigma_i}\right)^2

where :math:`x_i` is the model value, :math:`\mu_i` is the observed value, and
:math:`\sigma_i` is the uncertainty.

Optimizer
---------

The ``optimizer`` section configures the optimization algorithm.

Common options:

- ``algorithm``: Optimization algorithm (see below)
- ``maxiter``: Maximum number of iterations
- ``x0``: Optional initial guess as a list of parameter values. If not
  specified, local optimizers start from the center of the parameter bounds.

Supported algorithms:

- ``differential_evolution``: Global optimizer (recommended for initial
  exploration). Does not use ``x0``.
- ``nelder-mead``: Local optimizer (simplex method, derivative-free).
  Recommended for refining a previous result.
- ``powell``: Local optimizer (direction-set method)
- ``bfgs``: Local optimizer (gradient-based)
- ``cobyla``: Local optimizer (constrained)

Differential Evolution Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
        "optimizer": {
            "algorithm": "differential_evolution",
            "seed": 42,
            "maxiter": 1000,
            "tol": 0.01,
            "de_settings": {
                "strategy": "best1bin",
                "popsize": 15,
                "mutation": [0.5, 1.0],
                "recombination": 0.7,
                "workers": 1,
                "updating": "deferred",
                "polish": false,
                "disp": true
            }
        }
    }

.. warning::

    The ``polish`` option should be set to ``false`` for differential evolution
    to ensure the final solution respects parameter bounds.

Nelder-Mead Settings
^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
        "optimizer": {
            "algorithm": "nelder-mead",
            "maxiter": 5000,
            "x0": [3107.74, 4566.38, 0.06523, 8.808e-4, 2.354, 306026.0],
            "nm_settings": {
                "adaptive": true,
                "xatol": 1e-6,
                "fatol": 1e-6
            }
        }
    }

The ``nm_settings`` section supports:

- ``adaptive``: Use the adaptive algorithm that scales simplex operations
  based on dimensionality. Recommended for problems with more than 2
  parameters.
- ``xatol``: Absolute error in parameter values for convergence.
- ``fatol``: Absolute error in function value for convergence.
- ``initial_simplex``: Optional array of vertices to define the starting
  simplex.

.. note::

    Nelder-Mead does not enforce parameter bounds directly. The objective
    function returns ``failure_penalty`` for out-of-bounds evaluations,
    effectively constraining the search.

Output Settings
---------------

The ``output`` section configures result files:

.. code-block:: json

    {
        "output": {
            "results_file": "results.txt",
            "plot_evolution": false
        }
    }

- ``results_file``: Path to save optimization results
- ``plot_evolution``: Generate evolution plot (requires ``dStopTime`` parameter)
