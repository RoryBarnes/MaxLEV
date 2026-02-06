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

Install in development mode:

.. code-block:: bash

    pip install -e .

This installs all dependencies and registers the ``maxlev`` command.

Verify installation:

.. code-block:: bash

    maxlev --help

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
