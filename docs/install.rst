Installation Guide
==================

Prerequisites
-------------

``MaxLEV`` requires the following packages:

- Python 3.9 or higher
- ``vplanet`` and ``vplanet_inference``
- ``numpy``
- ``scipy``
- ``astropy``

Installing from Source
----------------------

Clone the repository:

.. code-block:: bash

    git clone https://github.com/RoryBarnes/MaxLEV.git
    cd MaxLEV

Install dependencies:

.. code-block:: bash

    pip install numpy scipy astropy vplanet vplanet_inference

Verify installation:

.. code-block:: bash

    python maxlev.py --help

VPLanet Configuration
---------------------

``MaxLEV`` requires a working ``VPLanet`` installation. You can specify the path to
the ``VPLanet`` executable in your configuration file:

.. code-block:: json

    {
        "vplanet": {
            "executable": "/path/to/vplanet"
        }
    }

If not specified, ``MaxLEV`` will use the ``vplanet`` command from your PATH.
