Examples
========

EarthInterior
-------------

This example demonstrates using ``MaxLEV`` to constrain Earth's initial thermal
state by fitting 6 parameters to 11 observational constraints using the
``thermint`` module.

The Problem
^^^^^^^^^^^

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

The optimization varies 6 parameters:

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

Running the Example
^^^^^^^^^^^^^^^^^^^

From the MaxLEV directory:

.. code-block:: bash

    python maxlev.py examples/EarthInterior/earthInterior.json

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

All observables are within 1.5 sigma of their observed values:

.. code-block:: text

    upperMantleTemperature         = 1.583259e+03 (obs: 1.587000e+03)
      Residual: -0.110 sigma
    innerCoreRadius                = 1.224059e+03 (obs: 1.224100e+03)
      Residual: -0.410 sigma

Generated Files
^^^^^^^^^^^^^^^

After optimization, the directory contains:

.. code-block:: text

    examples/EarthInterior/
        earth.in           # Original template
        earth_maxlev.in    # Generated with ML values
        sun.in
        vpl.in
    earthInterior_results.txt

The ``earth_maxlev.in`` file contains the maximum likelihood parameter values
and can be run directly with VPLanet:

.. code-block:: bash

    cd examples/EarthInterior
    vplanet vpl.in

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

Configuration File
^^^^^^^^^^^^^^^^^^

The complete configuration file is at ``examples/EarthInterior/earthInterior.json``.
