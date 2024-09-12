===========================================================================================================
FineST: Super resolved ligand-receptor interaction discovery by fusing spatial RNA-seq and histology images 
===========================================================================================================

About
=====

FineST (Fine_-grained S_ patial T_ ranscriptomics), a statistical model and toolbox to identify the super-resolved ligand-receptor interaction with spatial co-expression (i.e., spatial association). \

Uniquely, FineST can distinguish co-expressed ligand-receptor pairs (LR pairs) from spatially separating pairs at sub-spot level or single-cell level, and identify the super-resolved interaction.

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/fig/FineST_framework.png?raw=true
   :width: 900px
   :align: center

It comprises two main steps: \
1) Fine-grained ligand-receptor pair discovery; \
2) Cell-cell communication pattern classification; \
3) Pathway enrichment analysis.

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/fig/Downstream.png?raw=true
   :width: 900px
   :align: center

With the analytical testing method, FineST accurately predicts ST gene expression and outperforms TESLA and iStar at both spot and gene levels in terms of the root mean square error (RMSE) and Pearson correlation coefficient (PCC) between the predicted gene expressions and ground truth.

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/fig/OtherMethods.png?raw=true
   :width: 900px
   :align: center
   
It comprises two main steps: \
1) global selection `spatialdm_global` to identify significantly interacting LR pairs; \
2) local selection `spatialdm_local` to identify local spots for each interaction.

Installation
============

FineST is available through `PyPI <https://pypi.org/project/FineST/>`_. 
To install, type the following command line and add ``-U`` for updates:

.. code-block:: bash

   pip install -U FineST

Alternatively, you can install from this GitHub repository for latest (often 
development) version by the following command line:

.. code-block:: bash

   pip install -U git+https://github.com/LingyuLi-math/FineST

Installation time: < 1 min

Alternatively, 

.. code-block:: bash

   $ git clone https://github.com/LingyuLi-math/FineST.git
   $ conda create --name FineST python=3.8
   $ conda activate FineST
   $ pip install -r requirements.txt
Typically installation is expected to be completed within a few minutes. 


Quick example
=============

Using the build-in NPC dataset as an example, the following Python script
will predict super-resolution ST gene expression and compute the p-value indicating whether a certain Ligand-Receptor is 
spatially co-expressed. 


.. code-block:: python

        import spatialdm as sdm
        adata = sdm.datasets.dataset.melanoma()
        sdm.weight_matrix(adata, l=1.2, cutoff=0.2, single_cell=False) # weight_matrix by rbf kernel
        sdm.extract_lr(adata, 'human', min_cell=3)      # find overlapping LRs from CellChatDB
        sdm.spatialdm_global(adata, 1000, specified_ind=None, method='both', nproc=1)     # global Moran selection
        sdm.sig_pairs(adata, method='permutation', fdr=True, threshold=0.1)     # select significant pairs
        sdm.spatialdm_local(adata, n_perm=1000, method='both', specified_ind=None, nproc=1)     # local spot selection
        sdm.sig_spots(adata, method='permutation', fdr=False, threshold=0.1)     # significant local spots

        # visualize global and local pairs
        import spatialdm.plottings as pl
        pl.global_plot(adata, pairs=['SPP1_CD44'])
        pl.plot_pairs(adata, ['SPP1_CD44'], marker='s')
 
.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/figs/global_plot.png?raw=true
   :width: 200px
   :align: center
   
.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/figs/SPP1_CD44.png?raw=true
   :width: 600px
   :align: center



Detailed Manual
===============

The full manual is at https://finest-rtd-tutorial.readthedocs.io, including:  

* `Sub-spot level prediction (Recommended for small datasets, <2k spots)`_.

* `Sing-cell level prediction (Recommended for small datasets, <2k spots)`_.

.. _Sub-spot level prediction: tutorial/AEContraNPC1_16_LRgene_clear_0618pvalue.ipynb

.. _Sing-cell level prediction: tutorial/scAEContraNPC1_16_LRgene_clear_0604.ipynb




References
==========

SpatialDM manuscript with more details is available on bioRxiv_ now and is currently under review.

.. _bioRxiv: https://www.biorxiv.org/content/10.1101/2022.08.19.504616v1/

