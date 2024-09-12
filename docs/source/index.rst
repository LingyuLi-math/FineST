|PyPI| |Docs| |Build Status|

.. |PyPI| image:: https://img.shields.io/pypi/v/SpatialDM.svg
    :target: https://pypi.org/project/SpatialDM/
.. |Docs| image:: https://readthedocs.org/projects/spatialdm/badge/?version=latest
   :target: https://SpatialDM.readthedocs.io
.. |Build Status| image:: https://travis-ci.org/leeyoyohku/SpatialDM.svg?branch=main
   :target: https://travis-ci.org/leeyoyohku/SpatialDM


Home
====


About FineST
============

**FineST** (Fine-grained Spatial Transcriptomic) is a statistical model and toolbox to identify the super-resolved spatial co-expression (i.e., spatial association) between a pair of ligand and receptor.
It pulls data from the `Open Food Facts database <https://world.openfoodfacts.org/>`_
and offers a *simple* and *intuitive* API.

Uniquely, **FineST** can distinguish co-expressed ligand-receptor pairs (LR pairs) from spatially separating pairs at sub-spot level or single-cell level, and identify the super-resolved interaction.

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/fig/FineST_workflow.png?raw=true
   :width: 600px
   :align: center

Check out the :doc:`usage` section for further information, including
how to :ref:`installation` the project.

.. note::

   This project is under active development.

Please refer to our tutorials for details:

* `Sub-spot level prediction (Recommended for small datasets, <2k spots)`_.

* `Sing-cell level prediction`_.

.. _Sub-spot level prediction: tutorial/AEContraNPC1_16_LRgene_clear_0618pvalue.ipynb

.. _Sing-cell level prediction: tutorial/scAEContraNPC1_16_LRgene_clear_0604.ipynb


References
==========
FineST manuscript with more details is available on bioRxiv_ now and is currently under review.

.. _bioRxiv: https://www.biorxiv.org/content/10.1101/2022.08.19.504616v1/



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
