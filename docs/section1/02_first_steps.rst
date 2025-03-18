.. Intro to HPC @ TACC documentation master file, created by
   sphinx-quickstart on Fri Jun 26 14:44:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction to High Performance Computing
==========================================
What is High Performance Computing (HPC) ?
""""""""""""""""""""""""""""""""""""""""""
HPC refers to the aggregation of computing resources (like supercomputers or clusters) to achieve performance far greater than a single workstation or server

Basic HPC System Architecture
"""""""""""""""""""""""""""""
As you prepare to use TACC systems for this workshop, it is important to understand
the basic architecture. Think of an HPC resource as a very large and complicated lab
instrument. Users need to learn how to:

* Interface with it / push the right buttons (Linux)
* Load samples (data)
* Run experiments (jobs)
* Interpret the results (data analysis / vis)

.. image:: ./images/hpc_schematic.png
   :target: ./images/hpc_schematic.png
   :alt: HPC System Architecture

**Login vs. Compute Nodes**

As we've discussed, an HPC system has login nodes and compute nodes. We cannot run
applications on the login nodes because they require too many resources and will 
interrupt the work of others. Instead, we must submit a job to a queue to run on compute nodes.

Tips for Success
""""""""""""""""

Read the `documentation <https://docs.tacc.utexas.edu/>`_.

* Learn node schematics, limitations, file systems, rules
* Learn about the scheduler, queues, policies
* Determine the right resource for the job

User Responsibility on Shared Resources
"""""""""""""""""""""""""""""""""""""""

HPC systems are shared resources. Your jobs and activity on a cluster, if mismanaged,
can affect others. TACC staff are always `available to help <https://www.tacc.utexas.edu/about/help/>`_.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   steps_at_tacc/requesting
   steps_at_tacc/connecting
   steps_at_tacc/transferring
   steps_at_tacc/managing
   steps_at_tacc/running
   steps_at_tacc/understanding
   steps_at_tacc/executing

 