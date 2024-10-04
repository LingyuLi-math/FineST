.. automodule:: FineST

API
===

Import FineST as::
   import FineST  

Datasets
--------
The `FineST.datasets` module provides functions for loading and accessing spatial transcriptomics datasets. The following datasets are currently available:

* `dataset.NPC()`: Sample 1 rep 2 human melanoma slide from Thrane's melanoma dataset.
* `dataset.BRCA()`: Mouse sub-ventricular zone (SVZ) from Eng's seqfish+ dataset.

Usage
-----

To use the `FineST.datasets` module, simply import it as follows:

.. code-block:: python

    from FineST.datasets import dataset

Then, you can load a dataset using the corresponding function. For example, to load the melanoma dataset:

.. code-block:: python

    adata = dataset.melanoma()

This will return an `anndata` object containing the expression data for the melanoma dataset in `.X`, the cell type decomposition values in `.obs`, and the spatial coordinates in `.obsm['spatial']`.
