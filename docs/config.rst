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

The Gaussian likelihood assumes independent observables:

.. math::

    -\ln L = \frac{1}{2} \sum_i \left(\frac{x_i - \mu_i}{\sigma_i}\right)^2

where :math:`x_i` is the model value, :math:`\mu_i` is the observed value, and
:math:`\sigma_i` is the uncertainty.

Optimizer
---------

The ``optimizer`` section configures the optimization algorithm:

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

Supported algorithms:

- ``differential_evolution``: Global optimizer (recommended for most problems)
- ``powell``: Local optimizer
- ``nelder-mead``: Local optimizer (simplex method)
- ``bfgs``: Local optimizer (gradient-based)
- ``cobyla``: Local optimizer (constrained)

.. warning::

    The ``polish`` option should be set to ``false`` for differential evolution
    to ensure the final solution respects parameter bounds.

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
