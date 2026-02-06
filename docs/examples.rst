Examples
========

Each example uses the same VPLanet model of Earth's thermal interior but
demonstrates a different optimization algorithm. The science problem is
identical: constrain Earth's initial thermal state by fitting 6 parameters to 11
observational constraints using the ``thermint`` module.

The Problem
-----------

Earth's present-day thermal and magnetic properties provide constraints on its
initial conditions 4.5 billion years ago. The ``thermint`` module in VPLanet
simulates the coupled evolution of the mantle and core, producing outputs like:

- Upper mantle temperature and heat flow
- Core-mantle boundary temperature and heat flow
- Inner core radius
- Magnetic moment and magnetopause radius

Given modern measurements of these quantities, we can work backwards to find
the most likely initial conditions.

Parameters
^^^^^^^^^^

Both examples vary the same 6 parameters:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - Bounds
   * - ``dTMan``
     - Initial mantle temperature
     - 2500 - 3500 K
   * - ``dTCMB``
     - Initial CMB temperature
     - 4000 - 5000 K
   * - ``dEruptEff``
     - Melt eruption efficiency
     - 0.01 - 0.1
   * - ``dDTChiRef``
     - Core liquidus depression
     - 1e-4 - 1e-3 K
   * - ``dViscJumpMan``
     - Lower/upper mantle viscosity ratio
     - 1 - 5
   * - ``dActViscMan``
     - Viscosity activation energy
     - 1e5 - 5e5 J/mol

Observables
^^^^^^^^^^^

The model is constrained by 11 observables with uncertainties:

.. list-table::
   :header-rows: 1

   * - Observable
     - Value
     - Uncertainty
   * - Upper mantle temperature
     - 1587 K
     - +164/-34 K
   * - CMB temperature
     - 4000 K
     - 200 K
   * - Upper mantle heat flow
     - 38 TW
     - 3 TW
   * - CMB heat flow
     - 11 TW
     - 6 TW
   * - Upper mantle viscosity
     - 2.27e18 m2/s
     - 2.27e18 m2/s
   * - Lower mantle viscosity
     - 1.5e18 m2/s
     - 1.4e18 m2/s
   * - Upper mantle melt fraction
     - 0.115
     - 0.035
   * - Mantle melt mass flux
     - 1.3e6 kg/s
     - 0.8e6 kg/s
   * - Inner core radius
     - 1224.1 km
     - 0.1 km
   * - Magnetic moment
     - 80 ZA m^2
     - 4 ZA m^2
   * - Magnetopause radius
     - 9.1 R_Earth
     - 0.14 R_Earth

Unit Conversions
^^^^^^^^^^^^^^^^

Some VPLanet outputs require unit conversions. The configuration file specifies
``conversion_factor`` values for outputs that need scaling:

.. code-block:: json

    {
        "name": "final.earth.HflowUMan",
        "units": "TW",
        "conversion_factor": 1e-12,
        "description": "VPLanet reports in kg/sec^3; convert to TW"
    }

The magnetic moment and magnetopause radius are normalized to Earth's present
values for easier interpretation of the results.

Differential Evolution
----------------------

Differential evolution (DE) is a global optimizer that searches the entire
parameter space. It is the recommended algorithm for initial exploration of a
new problem because it does not depend on a starting point.

Configuration
^^^^^^^^^^^^^

The optimizer section uses ``differential_evolution`` with typical settings:

.. code-block:: json

    {
        "optimizer": {
            "algorithm": "differential_evolution",
            "seed": 42,
            "maxiter": 100,
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

    The ``polish`` option should be set to ``false`` to ensure the final
    solution respects parameter bounds.

Running the Example
^^^^^^^^^^^^^^^^^^^

From the MaxLEV directory:

.. code-block:: bash

    python maxlev.py examples/DifferentialEvolution/earthInterior.json

This runs differential evolution with 100 generations (approximately 1500
VPLanet simulations). Each simulation takes about 1 second, so the full
optimization takes approximately 30 minutes.

Results
^^^^^^^

The optimization produces:

.. code-block:: text

    EarthInterior Maximum Likelihood Estimation
    ======================================================================

    Maximum Likelihood Parameters:
    ----------------------------------------------------------------------
    earth.dTMan                    = 3.107743e+03
    earth.dTCMB                    = 4.566381e+03
    earth.dEruptEff                = 6.523120e-02
    earth.dDTChiRef                = 8.808236e-04
    earth.dViscJumpMan             = 2.354446e+00
    earth.dActViscMan              = 3.060260e+05

    -ln(Likelihood) = 3.039727e+00
    chi^2           = 6.079453e+00

With 11 observables and 6 parameters, there are 5 degrees of freedom. A chi^2
of 6.08 corresponds to a reduced chi^2 of 1.22, indicating a good fit.

The complete configuration file is at
``examples/DifferentialEvolution/earthInterior.json``.

Nelder-Mead
------------

The Nelder-Mead simplex method is a local optimizer that refines a solution from
a starting point. It is best used after differential evolution has identified
an approximate solution. Nelder-Mead is derivative-free, making it robust for
noisy or discontinuous objective functions.

Configuration
^^^^^^^^^^^^^

The optimizer section uses ``nelder-mead`` with algorithm-specific settings:

.. code-block:: json

    {
        "optimizer": {
            "algorithm": "nelder-mead",
            "maxiter": 5000,
            "nm_settings": {
                "adaptive": true,
                "xatol": 1e-6,
                "fatol": 1e-6
            }
        }
    }

The ``nm_settings`` section supports:

- ``adaptive``: Use the adaptive Nelder-Mead algorithm, which scales simplex
  operations based on dimensionality. Recommended for problems with more than
  2 parameters.
- ``xatol``: Absolute error in parameter values for convergence.
- ``fatol``: Absolute error in function value for convergence.
- ``initial_simplex``: Optional array of vertices to define the starting simplex.

To refine a previous differential evolution result, provide the DE solution as
the starting point with the ``x0`` option:

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

If ``x0`` is not specified, the optimizer starts from the center of the
parameter bounds.

Running the Example
^^^^^^^^^^^^^^^^^^^

From the MaxLEV directory:

.. code-block:: bash

    python maxlev.py examples/NelderMead/nelderMead.json

With ``adaptive`` enabled and 5000 maximum iterations, the optimization
typically converges in a few hundred function evaluations, taking a few minutes.

.. note::

    Nelder-Mead does not enforce parameter bounds directly. Instead, the
    objective function returns a large penalty value (``failure_penalty``) for
    any evaluation outside the bounds, effectively constraining the search.

The complete configuration file is at
``examples/NelderMead/nelderMead.json``.

Generated Files
---------------

After either optimization, the example directory contains:

.. code-block:: text

    examples/<Method>/
        earth.in           # Original template
        earth_maxlev.in    # Generated with ML values
        sun.in
        vpl.in
    <method>_results.txt

The ``earth_maxlev.in`` file contains the maximum likelihood parameter values
and can be run directly with VPLanet:

.. code-block:: bash

    cd examples/<Method>
    vplanet vpl.in
