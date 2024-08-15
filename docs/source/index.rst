Home
===================================

About FineST
===============

**FineST** (Fine-grained Spatial Transcriptomic) is a statistical model and toolbox to identify the super-resolved spatial co-expression (i.e., spatial association) between a pair of ligand and receptor.
It pulls data from the `Open Food Facts database <https://world.openfoodfacts.org/>`_
and offers a *simple* and *intuitive* API.

Uniquely, **FineST** can distinguish co-expressed ligand-receptor pairs (LR pairs) from spatially separating pairs at sub-spot level or single-cell level, and identify the super-resolved interaction.

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/figuse/FineST_workflow.png?raw=true
   :width: 600px
   :align: center


Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.


Please refer to our tutorials for details:

* `Permutation-based SpatialDM (Recommended for small datasets, <10k spots)`_.

* `Differential analyses of whole interactome among varying conditions`_.

.. _Permutation-based SpatialDM (Recommended for small datasets, <10k spots): melanoma.ipynb

.. _Differential analyses of whole interactome among varying conditions: differential_test_intestine.ipynb


Contents
==========

.. toctree::

   usage
   API

.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:
   
   index
   install
   quick_start
   api
   release

.. toctree::
   :caption: Examples
   :maxdepth: 1
   :hidden:

   melanoma
   differential_test_intestine


References
==========
FineST manuscript with more details is available on bioRxiv_ now and is currently under review.

.. _bioRxiv: https://www.biorxiv.org/content/10.1101/2022.08.19.504616v1/
