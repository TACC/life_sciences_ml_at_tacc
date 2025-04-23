 
Introduction to High Performance Computing
==========================================


What is High Performance Computing (HPC) ?
""""""""""""""""""""""""""""""""""""""""""
HPC refers to the aggregation of computing resources (like supercomputers or clusters) to achieve performance far greater than a single workstation or server


Basic HPC System Architecture
"""""""""""""""""""""""""""""
As you prepare to use TACC systems for this workshop, it is important to understand
the basic architecture. We can think of an HPC systeam as a very large and complicated laboratory
instrument. Users need to learn how to:

* Interface with it / push the right buttons (Linux)
* Load samples (data)
* Run experiments (jobs)
* Interpret the results (data analysis / vis)

.. image:: ./images/hpc_schematic.png
   :target: ./images/hpc_schematic.png
   :alt: HPC System Architecture
 
|

**IMPORTANT: Login vs. Compute Nodes**

An HPC system has login nodes and compute nodes. We cannot run
applications on the **login nodes** because they require too many resources and will 
interrupt the work of others. Instead, we must submit a job to a queue to run on **compute nodes**.
   
|

Main Systems available at TACC
""""""""""""""""""""""""""""""

Clusters
^^^^^^^^

`Frontera <https://tacc.utexas.edu/systems/frontera/>`_: The fastest academic supercomputer in the world, providing computational capability that makes larger, more complex research challenges possible.

`Vista <https://tacc.utexas.edu/systems/vista/>`_: Vista expands TACC’s capacity for AI and ensures that the broad science, engineering, and education research communities have access to the most advanced computing and AI technologies.

`Stampede3 <https://tacc.utexas.edu/systems/stampede3/>`_:
The newest strategic resource advancing NSF's supercomputing ecosystem for the nation's open science community.

`Lonestar6 <https://tacc.utexas.edu/systems/lonestar6/>`_:
Supporting Texas researchers in providing simulation, data analysis, visualization, and AI/machine learning.

`Jetstream2 <https://tacc.utexas.edu/systems/jetstream2/>`_:
A user-friendly, scalable cloud environment with reproducible, sharable computing on geographically isolated clouds.

Storage Systems
^^^^^^^^^^^^^^^

`Corral <https://tacc.utexas.edu/systems/corral/>`_:
Storage and data management resource designed and optimized to support large-scale collections and a collaborative research environment.

`Ranch <https://tacc.utexas.edu/systems/ranch/>`_:
Long-term data archiving environment designed, implemented, and supported to provide storage for data sets of the TACC user community.

`Stockyard <https://tacc.utexas.edu/systems/stockyard/>`_:
Global file system at the center of TACC's system ecosystem that supports data-driven science by providing online storage of large datasets, and offers migration for further data management and archiving.


File Systems
^^^^^^^^^^^^

The account-level environment variables $HOME, $SCRATCH and $WORK store the paths to directories that you own on each of these file systems. 
 
 

+---------------------+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
| File System         | Quota                             | Key Features                                                                                                       | 
+=====================+===================================+====================================================================================================================+
| ``$HOME``           |- 25GB                             |- Backed up.                                                                                                        |
|                     |- 200,000 files                    |- Recommended Use: scripts and templates, environment settings, compilation, cron jobs                              |
+---------------------+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``$WORK``           |- 1TB                              |- NOT backed up.                                                                                                    |
|                     |- 3,000,000 files                  |- Recommended Use: software installations, original datasets that can't be reproduced.                              |
+---------------------+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+
| ``$SCRATCH``        |- No quota assigned                |- NOT backed up.                                                                                                    |
|                     |                                   |                                                                                                                    |
|                     |                                   |- Recommended Use: Reproducible datasets, I/O files: temporary files, checkpoint/restart files, job output files    |
+---------------------+-----------------------------------+--------------------------------------------------------------------------------------------------------------------+

  
 


Files in $SCRATCH are subject to purge if access time is more than 10 days old.

A useful way to monitor your disk quotas (and TACC project balances) at any time is to execute:

.. code-block:: console

   [frontera]$ /usr/local/etc/taccinfo



