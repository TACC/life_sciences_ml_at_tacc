Running on HPC Systems
======================

Running tasks on a high performance computing (HPC) systems is different from running tasks on a
personal computer. You can either run tasks interactively on a compute node, or submit tasks as 
batch jobs to a queue. The former is useful for development (e.g. exploratory data analysis, testing
models, troubleshooting), while the latter is useful for production runs (e.g. training models at
scale). By the end of this section, you should be able to:

* Start an interactive session on a compute node
* Describe the components of a SLURM batch job submission file
* Use a Linux text editor such as VIM
* Submit a batch job to a queue
* Manage software modules


Run Interactively with idev
---------------------------

The ``idev`` utility initiates an interactive session on one or more compute nodes so that you can
issue commands and run code as if you were doing so on your personal machine. An interactive session
is useful for development, as you can quickly compile, run, and validate code. This is accomplished
by simply executing ``idev`` on any of the TACC systems.

To learn about the command line options available for ``idev``, use ``idev -help``.

.. code-block:: console
   
   [frontera]$ idev -help
   ...
   OPTION ARGUMENTS         DESCRIPTION
   -A     account_name      sets account name (default: in .idevrc)
   -m     minutes           sets time in minutes (default: 30)
   -p     queue_name        sets queue to named queue (default: -p development)
   -r     resource_name     sets hardware
   -t     hh:mm:ss          sets time to hh:mm:ss (default: 00:30:00)
   -help      [--help     ] displays (this) help message
   -v         [--version  ] output version information and exit

Let's go over some of the most useful ``idev`` command line options that can customize your
interactive session:

* To change the **time** limit from the default 30 minutes, users can apply the ``-m`` command line
  option. For example, a user requesting an interactive session for an hour would use the command
  line option ``-m 60``.
* To change the **allocation_name** associated with the interactive session, users can use the
  ``-A`` command line option. This option is useful for when a user has multiple allocations they
  belong to. For example, if I have allocations on accounts ``TACC`` and ``Training``, I can use
  ``-A`` to set the allocation I want to be used like so: ``-A TACC`` or ``-A Training``.
* To change the **queue** to be different than the default ``development`` queue, users can use the
  ``-p`` command line option. For example, if a user wants to launch an interactive session on one
  of Frontera GPU nodes, they would use the command line option ``-p rtx`` or ``-p rtx-dev``. You
  can learn more about the different queues of Frontera 
  `here <https://docs.tacc.utexas.edu/hpc/frontera/#table6>`_.

.. note::
   
   For the scope of this section, we will be using the default ``development`` queue.  

To start a thirty-minute interactive session on a compute node in the development queue with our
``EXAMPLE`` allocation:

.. code-block:: console
   
   [frontera]$ idev -A EXAMPLE   

If launch is successful, you will see output that includes the following excerpts:

.. code-block:: console
   
   ...
   -----------------------------------------------------------------
         Welcome to the Frontera Supercomputer          
   -----------------------------------------------------------------
   ...

   -> After your idev job begins to run, a command prompt will appear,
   -> and you can begin your interactive development session. 
   -> We will report the job status every 4 seconds: (PD=pending, R=running).

   -> job status:  PD
   -> job status:  R

   -> Job is now running on masternode= c205-004...OK
   ...
   c205-004[clx](633)$


EXERCISE
^^^^^^^^

Let's execute some Python code that determines the larger of two numbers. The code also includes a
3-second delay before finishing. Note, the ``[clx]`` syntax in the prompt refers to an interactive
session that has been started on a Frontera Cascade Lake node.

.. code-block:: console

   [clx]$ cdw
   [clx]$ unzip firststeps.zip
   [clx]$ cd Lab01
   [clx]$ pwd
   /work2/03302/lconcia/frontera/Lab01
   [clx]$ ls
   example.slurm  example_template.slurm  my_code.py

Load the appropriate modules, and run ``my_code.py``. 

.. code-block:: console

   [clx]$ module load python3
   [clx]$ python3 my_code.py
   The larger number of 51 and 20 is 51

You can check the files that were generated using ``ls``, and see the contents of the file with
``cat``.

.. code-block:: console

   [clx]$ ls
   duration.txt  example.slurm  example_template.slurm  my_code.py
   [clx]$ cat duration
   Done in 3.009739637374878 seconds.

To exit an interactive session, you can use the command ``logout``.

.. note::

   In case you didn't donwload the code in the previous section, you can download by doing:
   
   .. code-block:: console
   
      [frontera]$ cdw
      [frontera]$ wget https://github.com/TACC/life_sciences_ml_at_tacc/raw/refs/heads/main/docs/section1/files/firststeps.zip
      [frontera]$ unzip firststeps.zip


Run Non-Interactively with sbatch
---------------------------------

As we discussed before, on Frontera there are login nodes and compute nodes.

.. image:: ./images/hpc_schematic.png
   :target: ./images/hpc_schematic.png
   :alt: HPC System Architecture

We cannot run the applications we need for our research on the login nodes because they are designed
as a prep area, where you may edit and manage files, compile code, perform file management, issue
transfers, submit new and track existing batch jobs etc. The login nodes provide an interface to the
"back-end" compute nodes, where actual computations occur and where research is done. 

To run a job, instead, we must write a short text file containing a list of the resources we need,
and containing the job command(s). Then, we submit that text file to a queue to run on compute nodes.
This process is called **batch job submission**.

There are several queues available on Frontera. It is important to understand the queue limitations
and pick a queue that is appropriate for your job.  Documentation can be found
`here <https://docs.tacc.utexas.edu/hpc/frontera/#running-queues>`__. 
Today, we will be using the ``development`` queue which has a max runtime of 2 hours, and users can
only submit one job at a time.

First, navigate to the ``Lab01`` directory where we have an example job script prepared, called
``example_template.slurm``:

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
^^^^^^^^^^^^^^^^^^^^^^^^^^

Here, we are comparing the differences between two queues: ``development`` and ``normal``. 
For information about other queues, please refer to the
`Frontera Production Queues <https://docs.tacc.utexas.edu/hpc/frontera/#table6>`_.

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

.. note::

   If you submit a job requesting 48 hrs in the normal queue, and it takes a total of 10 hrs
   to run, you will be charged as follows:
   
   **SUs charged = (Number of nodes) X (job wall-clock time) X (charge rate per node-hour).**
   
   **SUs charged = (Number of nodes) X 10 X 1.**


GPUs Available at TACC 
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
| Stampede3                | Ponte Vecchio |      20         | 4x Intel Data Center Max 1550s  |   pvc                                           |
+--------------------------+---------------+-----------------+---------------------------------+-------------------------------------------------+
| Frontera                 |               |      90         | 4x NVIDIA Quadro RTX 5000       |   rtx                                           |
+                          +               +                 +                                 +-------------------------------------------------+
|                          |               |                 |                                 |   rtx-dev                                       |
+--------------------------+---------------+-----------------+---------------------------------+-------------------------------------------------+
| Vista                    | Grace Hopper  |      600        | 1x NVIDIA H200 GPU              |   gh                                            |
+                          +               +                 +                                 +-------------------------------------------------+
|                          |               |                 |                                 |   gh-dev                                        |
+--------------------------+---------------+-----------------+---------------------------------+-------------------------------------------------+


Executing Basic Job Management Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, we must know an application we want to run, and a research question we want to ask. In this
example, we aim to execute a Python code that determines the larger of two numbers. The code also
includes a 3-second delay before finishing.

.. code-block:: console

   [frontera]$ cdw
   [frontera]$ cd Lab01
   [frontera]$ pwd
   /work2/03302/lconcia/frontera/Lab01
   [frontera]$ ls
   example.slurm  example_template.slurm  my_code.py

Next, we need to fill out ``example_template.slurm`` to request the necessary resources. I know that
this code will take a little more than 3 seconds, so I can reasonably predict how much we will need. 
When running your first jobs with your applications, it will take some trial and error, and reading
online documentation, to get a feel for how many resources you should use. Open ``example_template.slurm``
with VIM and fill out the following information:

.. code-block:: console

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

   #SBATCH -J first_job                       # Job name
   #SBATCH -o output.%j                       # Name of stdout output file (%j expands to jobId)
   #SBATCH -e error.%j                        # Name of stderr error file (%j expands to jobId)
   #SBATCH -p development                     # Queue (partition) name
   #SBATCH -N 1                               # Total number of nodes (must be 1 for serial)
   #SBATCH -n 1                               # Total number of threas tasks requested (should be 1 for serial)
   #SBATCH -t 0:30:00                         # Run time (hh:mm:ss), development queue max 2:00:00
   #SBATCH -A Frontera-Training               # Project/Allocation name (req'd if you have more than 1)

   # Everything below here should be Linux commands

   module load python3

   python3 my_code.py

The way this job is configured, it will load the appropriate modules, and run ``my_code.py``. 


Text Editing with VIM
^^^^^^^^^^^^^^^^^^^^^

VIM is a text editor used on Linux file systems.

Open the file ``example_template.slurm``:

.. code-block:: console
 
   [frontera]$ vim example_template.slurm

There are two "modes" in VIM that we will talk about today. They are called "insert mode" and
"normal mode".  In insert mode, the user is typing text into a file as seen through the terminal
(think about typing text into TextEdit or Notepad).  In normal mode, the user can perform other
functions like save, quit, cut and paste, find and replace, etc. (think about clicking the menu
options in TextEdit or Notepad). The two main keys to remember to toggle between the modes are ``i``
and ``Esc``.

Entering VIM insert mode:

.. code-block:: bash

   > i

Entering VIM normal mode:

.. code-block:: bash

   > Esc

A summary of the most important keys to know for normal mode are (more on your cheat sheet):

.. code-block:: bash

   # Navigating the file:

   arrow keys        move up, down, left, right
       Ctrl+u        page up
       Ctrl+d        page down

            0        move to beginning of line
            $        move to end of line

           gg        move to beginning of file
            G        move to end of file
           :N        move to line N

   # Saving and quitting:

           :q        quit editing the file
           :q!       quit editing the file without saving

           :w        save the file, continue editing
           :wq       save and quit

For more information, see:
  * `http://openvim.com/ <http://openvim.com/>`_
  * Or type on the command line: ``vimtutor``


Submit a Batch Job to the Queue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have filled in the job description, save and quit the file. Submit the job to the queue
using the ``sbatch`` command`:

.. code-block:: console

   [frontera]$ sbatch example_template.slurm

To view the jobs you have currently in the queue, use the ``showq`` or ``squeue`` commands:

.. code-block:: console

   [frontera]$ showq -u               # shows my jobs
   [frontera]$ showq                  # shows all jobs by all users
   [frontera]$ squeue -u $USERNAME    # shows my jobs
   [frontera]$ squeue                 # shows all jobs by all users

If for any reason you need to cancel a job, use the ``scancel`` command with the 6- or 7-digit
``jobid``:

.. code-block:: console

   [frontera]$ scancel jobid

For more example scripts, see this directory on Frontera:

.. code-block:: console

   [frontera]$ ls /share/doc/slurm/

If everything went well, you should have a file named ``duration.txt``, 
an output file named something similar to ``output.o6146935``, 
and an error file named something similar to ``error.o6146935`` in the same directory as the
``example_template.slurm`` script. 

.. code-block:: console

   [frontera]$ ls
   duration.txt  error.6146935  example.slurm  example_template.slurm  my_code.py  output.6146935

**Congratulations! You ran a batch job on Frontera!**


Managing Modules
----------------

Modules enable users to run specific applications or access libraries without the need to log out
and back in. Modules for applications adjust the user's path for easy access, while those for
library packages set environment variables indicating the location of library and header files.
Switching between package versions or removing a package is straightforward.


Tour of the Module Command
^^^^^^^^^^^^^^^^^^^^^^^^^^

The module command sets the appropriate environment variable
independent of the user's shell.  Typically the system will load a
default set of modules.  A user can list the modules loaded by:

.. code-block:: console

   [frontera]$ module list
   
   Currently Loaded Modules:
     1) intel/19.1.1   3) git/2.24.1      5) python3/3.7.0   7) hwloc/1.11.12   9) TACC
     2) impi/19.0.9    4) autotools/1.2   6) cmake/3.24.2    8) xalt/2.10.34

To find out available modules for loading, a user can use:

.. code-block:: console

   [frontera]$ module avail
   - or -
   [frontera]$ module spider

Press the ``<Enter>`` key to scroll through line-by-line, or the ``<Space>`` key to scroll through
page-by-page. Press ``q`` to quit the view.

If there are many modules on a system, it can be difficult to see what
modules are available to load. To display a concise listing:

.. code-block:: console

   [frontera]$ module overview
   
   ----------------- /opt/apps/intel19/impi19_0/python3_7/modulefiles -----------------
   boost-mpi (2)
   
   --------------------- /opt/apps/intel19/python3_7/modulefiles ----------------------
   boost (2)
   
   ---------------------- /opt/apps/intel19/impi19_0/modulefiles ----------------------
   Rstats    (2)   gromacs    (4)    openfoam        (4)     remora      (2)
   adcirc    (1)   hpctoolkit (2)    opensees        (4)     rosetta     (1)
   adios2    (1)   hypre      (16)   p3dfft++        (1)     slepc       (39)
   amask     (1)   ipm        (1)    p4est           (2)     suitesparse (10)
   arpack    (1)   kokkos     (4)    parallel-netcdf (4)     sundials    (9)
   aspect    (1)   lammps     (4)    parmetis        (1)     superlu     (12)

This shows the short name of the module (i.e. git, or Rstats)
and the number in the parentheses is the number of versions for each.
This list above shows that there is one version of git and two
versions of Rstats.

To check all the versions of a package (e.g., Rstats):

.. code-block:: console

   [frontera]$ module avail Rstats
   
   --------------- /opt/apps/intel19/impi19_0/modulefiles ---------------
      Rstats/3.6.3    Rstats/4.0.3 (D)

In ``Rstats/4.0.3 (D)``, the ``(D)`` denotes the default module. When loading packages, if you don't
specify the version, the default module will be loaded. To load packages a user simply does:

.. code-block:: console

    [frontera]$ module load package1 package2 ...

To unload packages a user does:

.. code-block:: console

    [frontera]$ module unload package1 package2 ...

Modules can contain help messages.  To access a module's help do:

.. code-block:: console

    [frontera]$ module help packageName

To get a list of all the commands that module knows about do:

.. code-block:: console

    [frontera]$ module help


Review of Topics Covered
------------------------

**VIM**

+------------------------------------+-------------------------------------------------+
| Command                            |  Effect                                         |
+====================================+=================================================+
| ``vim file.txt``                   |  open "file.txt" and edit with ``vim``          |
+------------------------------------+-------------------------------------------------+
| ``i``                              |  toggle to insert mode                          |
+------------------------------------+-------------------------------------------------+
| ``<Esc>``                          |  toggle to normal mode                          |
+------------------------------------+-------------------------------------------------+
| ``<arrow keys>``                   |  navigate the file                              |
+------------------------------------+-------------------------------------------------+
| ``:q``                             |  quit ending the file                           |
+------------------------------------+-------------------------------------------------+
| ``:q!``                            |  quit editing the file without saving           |
+------------------------------------+-------------------------------------------------+
|  ``:w``                            |  save the file, continue editing                |
+------------------------------------+-------------------------------------------------+
|  ``:wq``                           |  save and quit                                  |
+------------------------------------+-------------------------------------------------+

**SLURM**

+------------------------------------+-------------------------------------------------+
| Command                            |  Effect                                         |
+====================================+=================================================+
| ``sbatch JOB.SLURM``               |  submit batch job to the queue                  |
+------------------------------------+-------------------------------------------------+
| ``showq``                          |  show all jobs by all users                     |
+------------------------------------+-------------------------------------------------+
| ``showq -u``                       |  show the current user's jobs                   |
+------------------------------------+-------------------------------------------------+
| ``squeue``                         |  show all jobs by all users                     |
+------------------------------------+-------------------------------------------------+
| ``squeue -u USERNAME``             |  show user USERNAME's jobs                      |
+------------------------------------+-------------------------------------------------+
| ``scancel JOBID``                  |  cancel a job given a job ID                    |
+------------------------------------+-------------------------------------------------+

**Modules**

+------------------------------------+-------------------------------------------------+
| Command                            | Effect                                          |
+====================================+=================================================+
| ``module list``                    | List currently loaded modules                   |
+------------------------------------+-------------------------------------------------+
| ``module avail``                   | See what modules are available                  |
+------------------------------------+-------------------------------------------------+
| ``module overview``                | See what modules are available (concise)        |
+------------------------------------+-------------------------------------------------+
| ``module avail name``              | Search for module "name"                        |
+------------------------------------+-------------------------------------------------+
| ``module load name``               | Load module "name"                              |
+------------------------------------+-------------------------------------------------+
| ``module unload name``             | Unload module "name"                            |
+------------------------------------+-------------------------------------------------+
| ``module help name``               | Show module "name" help                         |
+------------------------------------+-------------------------------------------------+
| ``module help``                    | Show module command help                        |
+------------------------------------+-------------------------------------------------+


Additional Resources
--------------------

* `Frontera queues <https://docs.tacc.utexas.edu/hpc/frontera/#table6>`_
* `VIM tutorial <http://openvim.com/>`_
* `Lmod module system <https://lmod.readthedocs.io/en/latest/>`_
