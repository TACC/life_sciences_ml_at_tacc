Understanding Slurm Job Submission Files
========================================

As we discussed before, on Frontera there are login nodes and compute nodes.

.. image:: ./images/hpc_schematic.png
   :target: ./images/hpc_schematic.png
   :alt: HPC System Architecture

We cannot run the applications we need for our research on the login nodes because they are designed as a prep area, where you may edit and manage files, compile code, perform file management, issue transfers, submit new and track existing batch jobs etc. The login nodes provide an interface to the "back-end" compute nodes, where actual computations occur and where research is done. 

To run a job, instead, we must write a short text file containing a list of the resources we need, and containing the job command(s). 
Then, we submit that text file to a queue to run on compute nodes. This process is called **batch job submission**.

There are several queues available on Frontera. It is important to understand the queue limitations and pick a queue that is appropriate for your job. 
Documentation can be found `here <https://docs.tacc.utexas.edu/hpc/frontera/#running-queues>`_. 
Today, we will be using the ``development`` queue which has a max runtime of 2 hours, and users can only submit one job at a time.

First, navigate to the ``Lab01`` directory where we have an example job script prepared, called ``example_template.slurm``:

.. code-block:: console

   [frontera]$ cdw
   [frontera]$ cd Lab01
   [frontera]$ cat example_template.slurm

   #!/bin/bash
   #----------------------------------------------------
   # Example SLURM job script to run applications on 
   # TACCs Frontera system.
   #
   # Example of job submission
   # To submit a batch job, execute:             sbatch example.slurm
   # To show all queued jobs from user, execute: showq -u
   # To kill a queued job, execute:              scancel <jobId>
   #----------------------------------------------------

   #SBATCH -J                                  # Job name
   #SBATCH -o                                  # Name of stdout output file (%j expands to jobId)
   #SBATCH -e                                  # Name of stderr error file (%j expands to jobId)
   #SBATCH -p                                  # Queue (partition) name
   #SBATCH -N                                  # Total number of nodes (must be 1 for serial)
   #SBATCH -n                                  # Total number of threas tasks requested (should be 1 for serial)
   #SBATCH -t                                  # Run time (hh:mm:ss), development queue max 2:00:00
   #SBATCH --mail-user=your_email@domaim.com   # Address email notifications
   #SBATCH --mail-type=all                     # Email at begin and end of job
   #SBATCH -A                                  # Project/Allocation name (req'd if you have more than 1)

   # Everything below here should be Linux commands

Frontera Production Queues
--------------------------

Here, we are comparing the differences between two queues: ``development`` and ``normal``. 
For information about other queues, please refer to the `Frontera Production Queues <https://docs.tacc.utexas.edu/hpc/frontera/#table6>`_.

.. table::
   :align: left
   :widths: auto

   ===================================== ======================== ==========================
   Queue Name                            ``development``          ``normal``
   ===================================== ======================== ==========================
   Min-Max Nodes per Job (assoc'd cores) 1-40 nodes (2,240 cores) 3-512 nodes (28,672 cores)
   Max Job Duration                      2 hrs                    48 hrs
   Max Nodes per User                    40 nodes                 1836 nodes
   Max Jobs per User                     1 job                    100 jobs
   Charge Rate per node-hour             1 SU                     1 SU 
   ===================================== ======================== ==========================

**Note**: If you submit a job requesting 48 hrs in the normal queue, and it takes a total of 10 hrs to run, you will be charged as follows:

**SUs charged = (Number of nodes) X (job wall-clock time) X (charge rate per node-hour).**

**SUs charged = (Number of nodes) X 10 X 1.**



GPUs available at TACC 
^^^^^^^^^^^^^^^^^^^^^^

Users frequently need to access GPUs to accelerate their machine learning workloads. 
Here a summary of GPUs available at TACC.

+--------------------------+---------------+-----------------+---------------------------------+-------------------------------------------------+
| System                   | GPU Nodes     |         #       |      GPUs per node              |     Queues                                      |
+==========================+===============+=================+=================================+=================================================+
| Lonestar6                |   A100        |    84           | 3x NVIDIA A100                  | gpu-a100                                        |
+                          +               +                 +                                 +-------------------------------------------------+
|                          |               |                 |                                 | gpu-a100-dev                                    |
+                          +               +                 +                                 +-------------------------------------------------+
|                          |               |                 |                                 | gpu-a100-small                                  |
+                          +---------------+-----------------+---------------------------------+-------------------------------------------------+
|                          |   H100        |       4         | 2x NVIDIA H100                  | gpu-h100                                        |
+--------------------------+---------------+-----------------+---------------------------------+-------------------------------------------------+
| Stampede3                | Ponte Vecchio |      20         | 4x Intel Data Center Max 1550s  |     pvc                                         |
+--------------------------+---------------+-----------------+---------------------------------+-------------------------------------------------+
| Frontera                 |               |      90         | 4x NVIDIA Quadro RTX 5000       |   rtx                                           |
+                          +               +                 +                                 +-------------------------------------------------+
|                          |               |                 |                                 |   rtx-dev                                       |
+--------------------------+---------------+-----------------+---------------------------------+-------------------------------------------------+
| Vista                    | Grace Hopper  |      600        | 1x NVIDIA H200 GPU              |   gh                                            |
+                          +               +                 +                                 +-------------------------------------------------+
|                          |               |                 |                                 |   gh-dev                                        |
+--------------------------+---------------+-----------------+---------------------------------+-------------------------------------------------+
 
 