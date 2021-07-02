Welcome to som-pbc's documentation!
===================================

**som-pbc**

.. image:: https://img.shields.io/pypi/v/som-pbc.svg
   :target: https://pypi.org/project/som-pbc/

**Authors:** Alex Müller

**Copyright:** (c) 2017 - 2021; Alex Müller

This package contains a simple self-organizing map implementation in Python
with periodic boundary conditions.

Self-organizing maps are also called Kohonen maps and were invented by
Teuvo Kohonen.[1] They are an unsupervised machine learning technique to
efficiently create spatially organized internal representations of
various types of data. For example, SOMs are well-suited for the
visualization of high-dimensional data.

This is a simple implementation of SOMs in Python. This SOM has periodic
boundary conditions and therefore can be imagined as a "donut". The
implementation uses ``numpy``, ``scipy``, ``scikit-learn`` and
``matplotlib``.

The project's GitHub page can be found here: http://github.com/alexarnimueller/som

.. toctree::
   :caption: The package documentation has the following contents:
   :maxdepth: 2
   :numbered:

   readme
   som
   examples
   license
   todo


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`