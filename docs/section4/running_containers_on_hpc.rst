Running Containers on HPC Systems
=================================

High performance computing (HPC) systems serve a large role in academic computing at scale.
In this portion of the training, we will explore methods for running containers that you develop
on HPC systems and also discovering containers built by the community that you can utilize. After
going through this section, you will be able to:

- Use Apptainer to execute Docker containers on a HPC system
- Understand how to run containers that use GPUs for computation
- Discover community curated software containers available at TACC

Introduction to Apptainer
-------------------------

.. Note::

    Prerequisites
	This section uses the Frontera compute cluster to run Apptainer. An active allocation on Frontera is required, though most content will apply to any system that supports Apptainer.

At face value, Apptainer is an alternative container implementation to Docker that has an overlapping
set of features but some key differences as well.  Apptainer is commonly available on shared clusters,
such as TACC's HPC systems, because the Docker runtime is not secure on systems where users are not
allowed to have "escalated privileges".  Importantly, the Apptainer runtime is compatible with Docker
containers!  So in general, we follow the practice of using Docker to develop containers and using
Apptainer simply as a runtime to execute containers on HPC systems.

If you are familiar with Docker, Apptainer will feel familiar.


Login to Frontera
~~~~~~~~~~~~~~~~~

For today's training, we will use the Frontera supercomputer, the 33rd most powerful system in the world at the time of the course.  To login, you need to establish a SSH connection from your laptop to the Frontera system.  Instructions depend on your laptop's operating system.

Mac / Linux:

|   Open the application 'Terminal'
|   ssh username@frontera.tacc.utexas.edu
|   (enter password)
|   (enter 6-digit token)


Windows:

|   If using Windows Subsystem for Linux, use the Mac / Linux instructions.
|   If using an application like 'PuTTY'
|   enter Host Name: frontera.tacc.utexas.edu
|   (click 'Open')
|   (enter username)
|   (enter password)
|   (enter 6-digit token)


When you have successfully logged in, you should be greeted with some welcome text and a command prompt.


Start an Interactive Session
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Apptainer module is currently only available on compute nodes at TACC. To use Apptainer interactively,
start an interactive session on a compute node using the ``idev`` command.

.. code-block:: console

	$ idev -m 40

If prompted to use a reservation, choose yes.  Once the command runs successfully, you will no longer be
on a login node, but instead have a shell on a dedicated compute node.


Load the Apptainer Module
~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the ``apptainer`` command is not visible, but it can be added to the environment by loading
the module.

.. code-block:: console

	$ module list

	$ module spider apptainer

	$ module load tacc-apptainer

	$ module list

Now the apptainer command is available.

.. code-block:: console

	$ type apptainer

	$ apptainer help


Core Apptainer Commands
~~~~~~~~~~~~~~~~~~~~~~~


Pull a Docker container
^^^^^^^^^^^^^^^^^^^^^^^

Containers in the Docker registry may be downloaded and used, assuming the underlying
architecture (e.g. x86) is the same between the container and the host.

.. code-block:: console

	$ apptainer pull docker://godlovedc/lolcow

	$ ls

There may be some warning messages, but this command should download the latest version of the
"lolcow" container and save it in your current working directory as ``lolcow_latest.sif``.


Interactive shell
^^^^^^^^^^^^^^^^^

The ``shell`` command allows you to spawn a new shell within your container and interact with it
as though it were a small virtual machine.

.. code-block:: console

	$ apptainer shell lolcow_latest.sif

	Apptainer>

The change in prompt indicates that you have entered the container (though you should not rely on that
to determine whether you are in container or not).

Once inside of an Apptainer container, you are the same user as you are on the host system.
Also, a number of host directories are mounted by default.

.. code-block:: bash

	Apptainer> whoami

	Apptainer> id

	Apptainer> pwd

	Apptainer> exit


.. Note::

	Docker and Apptainer have very different conventions around how host directories are mounted within the container. In many ways, Apptainer has a simpler process for working with data on the host, but it is also more prone to inadvertantly having host configurations "leak" into the container.


Run a container's default command
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Just like with Docker, Apptainer can run the default "entrypoint" or default command of a container with
the ``run`` subcommand.  These defaults are defined in the Dockerfile (or Apptainer Definition file) that
define the actions a container should perform when someone runs it.

.. code-block:: console

	$ apptainer run lolcow_latest.sif

     ________________________________________
    < The time is right to make new friends. >
     ----------------------------------------
            \   ^__^
             \  (oo)\_______
                (__)\       )\/\
                    ||----w |
                    ||     ||


.. Note::

    You may receive a warning about "Setting locale failed".  This is because, by default, Apptainer sets all shell environment variables inside the container to match whatever is on the host. To override this behavior, add the ``--cleanenv`` argument to your command.


Executing arbitrary commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The exec command allows you to execute a custom command within a container. For instance, to execute
the ``cowsay`` program within the lolcow_latest.sif container:

.. code-block:: console

	$ apptainer exec --cleanenv lolcow_latest.sif cowsay Apptainer runs Docker containers on HPC systems
     _______________________________________
    / Apptainer runs Docker containers on \
    \ HPC systems                           /
     ---------------------------------------
            \   ^__^
             \  (oo)\_______
                (__)\       )\/\
                    ||----w |
                    ||     ||

.. Note::

	``exec`` also works with the library://, docker://, and shub:// URIs. This creates an ephemeral container that executes a command and disappears.

Once you are finished with your interactive session, you can end it and return to the login node with
the exit command:

.. code-block:: console

	$ exit


Apptainer in HPC Environments
-----------------------------

Conducting analyses on high performance computing clusters happens through very different patterns of
interaction than running analyses on a VM or on your own laptop.  When you login, you are on a node
that is shared with lots of people.  Trying to run jobs on that node is not "high performance" at all.
Those login nodes are just intended to be used for moving files, editing files, and launching jobs.

Most jobs on a HPC cluster are neither interactive, nor realtime.  When you submit a job to the scheduler,
you must tell it what resources you need (e.g. how many nodes, what type of nodes) and what you want to run.
Then the scheduler finds resources matching your requirements, and runs the job for you when it can.

For example, if you want to run the command:

.. code-block:: text

  apptainer exec docker://python:latest /usr/local/bin/python --version

On a HPC system, your job submission script would look something like:

.. code-block:: bash

  #!/bin/bash

  #SBATCH -J myjob                             # Job name
  #SBATCH -o output.%j                         # Name of stdout output file (%j expands to jobId)
  #SBATCH -p rtx                               # Queue name
  #SBATCH -N 1                                 # Total number of nodes requested (56 cores/node)
  #SBATCH -n 1                                 # Total number of mpi tasks requested
  #SBATCH -t 02:00:00                          # Run time (hh:mm:ss) - 4 hours
  #SBATCH --reservation <my_reservation>       # a reservation only active during the training

  module load tacc-apptainer
  apptainer exec docker://python:latest /usr/local/bin/python --version

This example is for the Slurm scheduler, a popular one used by all TACC systems.  Each of the #SBATCH lines
looks like a comment to the bash kernel, but the scheduler reads all those lines to know what resources
to reserve for you.

.. Note::

  Every HPC cluster is a little different, but they almost universally have a "User's Guide" that serves both as a quick reference for helpful commands and contains guidelines for how to be a "good citizen" while using the system.  For TACC's Frontera system, the user guide is at: `https://frontera-portal.tacc.utexas.edu/user-guide/ <https://frontera-portal.tacc.utexas.edu/user-guide/>`_


How do HPC systems fit into the development workflow?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


A couple of things to consider when using HPC systems:

#. Using 'sudo' is not allowed on HPC systems, and building an Apptainer container from scratch requires sudo.  That means you have to build your containers on a different development system, which is why we started this course developing Docker on your own laptop).  You can pull a docker image on HPC systems.
#. If you need to edit text files, command line text editors don't support using a mouse, so working efficiently has a learning curve.  There are text editors that support editing files over SSH.  This lets you use a local text editor and just save the changes to the HPC system.

In general, most TACC staff that work with containers develop their code locally and then deploy their
containers to HPC systems to do analyses at scale.  If the containers are written in a way that
accommodates the small differences between the Docker and Apptainer runtimes, the transition is fairly
seamless.

Differences between Docker and Apptainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Host Directories
^^^^^^^^^^^^^^^^

**Docker:** None by default. Use ``-v <source>:<destination>`` to mount a source host directory to an arbitrary destination within the container.

**Apptainer:** Mounts your current working directory, $HOME directory, and some system directories by default. Other defaults may be set in a system-wide configuration. The ``--bind`` flag is supported but rarely used in practice.

User ID
^^^^^^^

**Docker:** Defined in the Dockerfile, but containers run as root unless a different user is defined or specified on the command line.  This user ID only exists within the container, and care must be taken when working with files on the host filesystem to make sure permissions are set correctly.

**Apptainer:** Containers are run in "userspace", so you are the same user and user ID both inside and outside the container.

Image Format
^^^^^^^^^^^^

**Docker:** Containers are stored in layers and managed in a repository by Docker.  The ``docker images`` command will show you what containers are on your local machine and images are always referenced by their repository and tag name.

**Apptainer:** Containers are files.  Apptainer can build a container on the fly if you specify a repository, but ultimately they are stored as individual files, with all the benefits and dangers inherent to files.


Running a Batch Job on Frontera
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are not already, please login to the Frontera system, just like we did at the start of the
previous section.  You should be on one of the login nodes of the system.

We will not be editing much text directly on Frontera, but we need to do a little.  If you have a text
editor you prefer, use it for this next part.  If not, the ``nano`` text editor is probably the most
accessible for those new to Linux.

Create a file called "pi.slurm" on the work filesystem:

.. code-block:: console

  $ cd $WORK
  $ mkdir life-sciences-ml-at-tacc
  $ cd life-sciences-ml-at-tacc
  $ nano classify.slurm

Those commands should open a new file in the nano editor.  Either type in (or copy and paste) the
following Slurm script.

.. code-block:: bash

  #!/bin/bash

  #SBATCH -J classify-image                    # Job name
  #SBATCH -o output.%j                         # Name of stdout output file (%j expands to jobId)
  #SBATCH -p rtx                               # Queue name
  #SBATCH -N 1                                 # Total number of nodes requested (56 cores/node)
  #SBATCH -n 1                                 # Total number of mpi tasks requested
  #SBATCH -t 00:10:00                          # Run time (hh:mm:ss)
  #SBATCH --reservation <my_reservation>       # a reservation only active during the training

  module load tacc-apptainer

  cd $SCRATCH

  echo "running the lolcow container:"
  apptainer run docker://godlovedc/lolcow:latest

  echo "grabbing image dog.jpg:"
  wget https://raw.githubusercontent.com/TACC/life_sciences_ml_at_tacc/main/docs/images/dog.jpg

  echo "classify image dog.jpg:"
  apptainer exec --nv docker://USERNAME/image-classifier:0.1 image_classifier.py dog.jpg

* Don't forget to replace ``USERNAME`` with your DockerHub username! If you didn't publish an image-classifier container from the previous sections, you are welcome to use "eriksf" as the username to pull my container.

* If you have more than one allocation, you will need to add another line specifying what allocation to use, such as: ``#SBATCH -A AllocationName``

Once you are done, try submitting this file as a job to Slurm.

.. code-block:: console

  $ sbatch classify.slurm

You can check the status of your job with the command ``showq -u``.

Once your job has finished, take a look at the output:

.. code-block:: console

  $ cat output*

Apptainer and GPU Computing
---------------------------

Apptainer **fully** supports GPU utilization by exposing devices at runtime with the ``--nv`` flag.
This is similar to ``nvidia-docker``, so all docker containers with libraries that are compatible with
the drivers on our systems can work as expected.

As a base, we recommend starting with the official CUDA
(`nvidia/cuda <https://hub.docker.com/r/nvidia/cuda>`_) images from NVIDIA on Docker Hub.  If you
specifically want to use `PyTorch <https://pytorch.org/>`_ or `Tensorflow <https://www.tensorflow.org/>`_
then the official repositories on Docker Hub, `pytorch/pytorch <https://hub.docker.com/r/pytorch/pytorch>`_
and `tensorflow/tensorflow <https://hub.docker.com/r/tensorflow/tensorflow>`_ respectively, are good
starting points.

Alternatively, the `NVIDIA GPU Cloud <https://ngc.nvidia.com/>`_ (NGC) has a large number of pre-built
containers for deep learning and HPC applications including
`PyTorch <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_ and
`Tensorflow <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow>`_ (full-featured, large,
and include ARM64 versions). 

For instance, the latest version of Caffe can be used on TACC systems as follows:

.. code-block:: console

  Work from a compute node
  $ idev -m 60 -p rtx

  Load the apptainer module
  $ module load tacc-apptainer

  Pull your image
  $ apptainer pull docker://nvidia/caffe:latest

  Test the GPU
  $ apptainer exec --nv caffe_latest.sif caffe device_query -gpu 0

.. Note::

	If this resulted in an error and the GPU was not detected, and you are on a GPU-enabled compute node, you may have excluded the ``--nv`` flag.

As previously mentioned, the main requirement for GPU-enabled containers to work is that the version of the
NVIDIA host driver on the system supports the version of the CUDA library inside the container.

For some more exciting examples, lets look at two of the most popular Deep Learning frameworks for
Python, `Tensorflow <https://www.tensorflow.org/>`_ and `PyTorch <https://pytorch.org/>`_.

First, we'll run a simple script (`tf_test.py <https://raw.githubusercontent.com/TACC/life_sciences_ml_at_tacc/main/docs/scripts/tf_test.py>`_)
that uses Tensorflow to show the GPUs and then creates two tensors and multiplies them together.
It can be tested as follows:

.. code-block:: console

  Change to your $SCRATCH directory
  $ cd $SCRATCH

  Download the test code
  $ wget https://raw.githubusercontent.com/TACC/life_sciences_ml_at_tacc/main/docs/scripts/tf_test.py

  Pull the image
  $ apptainer pull docker://tensorflow/tensorflow:latest-gpu

  Run the code
  $ apptainer exec --nv tensorflow_latest-gpu.sif python tf_test.py 2>warnings.txt
  Tensorflow version: 2.17.0
  GPU available: True

  GPUs:
  Name: /physical_device:GPU:0   Type: GPU
  Name: /physical_device:GPU:1   Type: GPU
  Name: /physical_device:GPU:2   Type: GPU
  Name: /physical_device:GPU:3   Type: GPU

  TNA= tf.Tensor(
  [[1. 2. 3.]
  [4. 5. 6.]], shape=(2, 3), dtype=float32)
  TNB= tf.Tensor(
  [[1. 2.]
  [3. 4.]
  [5. 6.]], shape=(3, 2), dtype=float32)
  TNAxTNB= tf.Tensor(
  [[22. 28.]
  [49. 64.]], shape=(2, 2), dtype=float32)

.. Note::

	If you would like avoid the wordy tensorflow warning messages, run the above command and
	redirect STDERR to a file (i.e. ``2>warnings.txt``).

Next, we'll look at another example of matrix multiplication using PyTorch (`pytorch_matmul_scaling_test.py <https://raw.githubusercontent.com/TACC/life_sciences_ml_at_tacc/main/docs/scripts/pytorch_matmul_scaling_test.py>`_)
where we'll show how long it takes to multiply increasingly bigger matrices using both the CPU and GPU.
It can be tested as follows:

.. code-block:: console

  Change to your $SCRATCH directory
  $ cd $SCRATCH

  Download the test code
  $ wget https://raw.githubusercontent.com/TACC/life_sciences_ml_at_tacc/main/docs/scripts/pytorch_matmul_scaling_test.py

  Pull the image
  $ apptainer pull docker://eriksf/pytorch:2.4.1-cuda12.1-cudnn9-runtime

  Run the code against the CPU
  $ apptainer exec --nv pytorch_2.4.1-cuda12.1-cudnn9-runtime.sif python pytorch_matmul_scaling_test.py --no-gpu
  INFO:    gocryptfs not found, will not be able to use gocryptfs
  PyTorch Matrix Multiplication Test for Large Matrices
  PyTorch version: 2.4.1+cu121
  Using device: cpu

  Running test for matrix size: 2048x2048
  Estimated memory requirement: 0.03 GB

  Running test for matrix size: 4096x4096
  Estimated memory requirement: 0.12 GB

  Running test for matrix size: 8192x8192
  Estimated memory requirement: 0.50 GB
                   Matrix Multiplication Test Results
  ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
  ┃ Matrix Size ┃ Memory Size (GB) ┃ Computation Time (s) ┃ Performance  ┃
  ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
  │ 2048x2048   │ 0.03             │ 0.1995               │ 86.13 GFLOPS │
  │ 4096x4096   │ 0.12             │ 1.5595               │ 88.13 GFLOPS │
  │ 8192x8192   │ 0.50             │ 12.4375              │ 88.40 GFLOPS │
  └─────────────┴──────────────────┴──────────────────────┴──────────────┘
  Scaling plot saved as 'scaling_plot.png'

The script also produces a scaling plot:

.. figure:: ../images/scaling_plot_cpu.png
  :align: center

  Scaling plot for CPU

.. code-block:: console

  Run the code against the GPU
  $ apptainer exec --nv pytorch_2.4.1-cuda12.1-cudnn9-runtime.sif python pytorch_matmul_scaling_test.py
  INFO:    gocryptfs not found, will not be able to use gocryptfs
  PyTorch Matrix Multiplication Test for Large Matrices
  PyTorch version: 2.4.1+cu121
  Using device: cuda
  CUDA version: 12.1
  GPU: Quadro RTX 5000
  GPU Memory: 15.74 GB

  Running test for matrix size: 2048x2048
  Estimated memory requirement: 0.03 GB

  Running test for matrix size: 4096x4096
  Estimated memory requirement: 0.12 GB

  Running test for matrix size: 8192x8192
  Estimated memory requirement: 0.50 GB
                  Matrix Multiplication Test Results
  ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
  ┃ Matrix Size ┃ Memory Size (GB) ┃ Computation Time (s) ┃ Performance ┃
  ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
  │ 2048x2048   │ 0.03             │ 0.0037               │ 4.67 TFLOPS │
  │ 4096x4096   │ 0.12             │ 0.0288               │ 4.78 TFLOPS │
  │ 8192x8192   │ 0.50             │ 0.2252               │ 4.88 TFLOPS │
  └─────────────┴──────────────────┴──────────────────────┴─────────────┘
  Scaling plot saved as 'scaling_plot.png'

The script also produces a scaling plot:

.. figure:: ../images/scaling_plot.png
  :align: center

  Scaling plot for GPU


Building a GPU aware container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
