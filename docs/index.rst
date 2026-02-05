MaxLEV Documentation
=====================
``MaxLEV`` (Maximum Likelihood Estimation for VPLanet) is a tool for finding the
maximum likelihood parameters of ``VPLanet`` simulations given observational constraints.

With ``MaxLEV`` you can define optimization parameters, observational constraints with
uncertainties (including asymmetric error bars), and use differential evolution or other
scipy optimizers to find the best-fit parameters. ``MaxLEV`` performs comprehensive
validation of all parameters, outputs, and body names against ``VPLanet``'s actual options
before running any simulations.

After optimization, ``MaxLEV`` generates:

- A results file with best-fit parameters, chi-squared, and residuals for each observable
- Modified ``VPLanet`` input files (``*_maxlev.in``) containing the maximum likelihood values

.. toctree::
   :maxdepth: 1

   install
   help
   config
   examples
   GitHub <https://github.com/RoryBarnes/MaxLEV>

.. note::

    ``MaxLEV`` is designed to work with ``vplanet_inference`` for running ``VPLanet``
    simulations. For posterior inference (beyond point estimates), consider using
    `alabi <https://github.com/VirtualPlanetaryLaboratory/alabi>`_ after identifying
    the maximum likelihood solution.
