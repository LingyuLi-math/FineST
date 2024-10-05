===========================================
FineST: Fine-grained Spatial Transcriptomic
===========================================

About
=====

FineST (Fine-grained Spatial Transcriptomics), a statistical model and toolbox to identify the super-resolved ligand-receptor interaction with spatial co-expression (i.e., spatial association).

Uniquely, FineST can distinguish co-expressed ligand-receptor pairs (LR pairs) from spatially separating pairs at sub-spot level or single-cell level, and identify the super-resolved interaction.

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/fig/FineST_framework.png?raw=true
   :width: 600px
   :align: center

It comprises two main steps:

1. Fine-grained ligand-receptor pair discovery;
2. Cell-cell communication pattern classification;
3. Pathway enrichment analysis.

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/fig/Downstream.png?raw=true
   :width: 600px
   :align: center

With the analytical testing method, FineST accurately predicts ST gene expression and outperforms TESLA and iStar at both spot and gene levels in terms of the root mean square error (RMSE) and Pearson correlation coefficient (PCC) between the predicted gene expressions and ground truth.

.. image:: https://github.com/LingyuLi-math/FineST/blob/main/docs/fig/OtherMethods.png?raw=true
   :width: 600px
   :align: center

It comprises two main steps:

1. global selection `spatialdm_global` to identify significantly interacting LR pairs;
2. local selection `spatialdm_local` to identify local spots for each interaction.

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
   $ cd FineST
   $ pip install -r requirements.txt

Typically installation is expected to be completed within a few minutes. 
Then install pytorch. You may refer to `pytorch installation <https://pytorch.org/get-started/locally/>`_.

.. code-block:: bash

   $ conda install pytorch=1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch

Once the installation is complete, you can verify the installation using the following command:

.. code-block:: bash

   python
   >>> import torch
   >>> print(torch.__version__)
   >>> print(torch.cuda.is_available())

Quick example
=============

Using the build-in NPC dataset as an example, the following Python script
will predict super-resolution ST gene expression and compute the p-value indicating whether a certain Ligand-Receptor is
spatially co-expressed.

Detailed Manual
===============

The full manual is at `finest-rtd-tutorial <https://finest-rtd-tutorial.readthedocs.io>`_.