Set up for Tutorials
====================

This section provides instructions for setting up the environment and gathering the correct files 
for two hands-on tutorials.


Set Up For Frontera
------------------- 


Step 1. Log in to Frontera
^^^^^^^^^^^^^^^^^^^^^^^^^^

Log in to Frontera using SSH:

.. code:: console

   [local]$ ssh username@frontera.tacc.utexas.edu
   (username@frontera.tacc.utexas.edu) Password: 
   (username@frontera.tacc.utexas.edu) TACC Token Code:

   # ------------------------------------------------------------------------------
   # Welcome to the Frontera Supercomputer
   # Texas Advanced Computing Center, The University of Texas at Austin
   # ------------------------------------------------------------------------------


Step 2. Gather the Tutorial Materials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to your scratch directory and clone this tutorial repository:

.. code:: console

   [frontera]$ cds # shortcut for cd #SCRATCH
   [frontera]$ git clone https://github.com/kbeavers/tacc-deep-learning-tutorials


Step 3. Set up Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Start an interactive session on a development node, then run the setup script.

.. code:: console

   [frontera]$ cds
   [frontera]$ idev -m 20
   ...
   [clx]$ # You are now in an interactive session on a compute node

.. code:: console

   # Load the Apptainer module
   [clx]$ module load tacc-apptainer

   # Pull the Docker container image created for this tutorial
   [clx]$ apptainer pull docker://kbeavers/tf-213:frontera

   # Run the kernel setup script
   [clx]$ cd tacc-deep-learning-tutorials/
   [clx]$ bash ./scripts/install_kernel.sh


Step 4. Dataset Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extract the provided coral species image dataset.

.. code:: console

   [clx]$ bash ./scripts/download_dataset.sh


Step 5. Copy the Notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^

Copy the tutorial notebooks to your home directory. These notebooks are provided as blank templates
for you to fill in as you work through the exercises.

.. code:: console

   [clx]$ cp ./tutorials/Mushroom-ANN-tutorial.ipynb $HOME/
   [clx]$ cp ./tutorials/Coral-CNN-tutorial.ipynb $HOME/

At this point, you should log out of the interactive session.

.. code:: console

   [clx]$ logout
   ...
   [frontera]$ # You are now back on the Frontera login node


Step 6. Launch Jupyter
^^^^^^^^^^^^^^^^^^^^^^

Log in to the `TACC Analysis Portal <https://tap.tacc.utexas.edu/jobs/>`_ and configure your session
as follows:

* **System:** Frontera
* **Application:** Jupyter Notebook
* **Project:** Frontera-Training
* **Queue:** rtx
* **Job Name:** DL-Training
* **Time Limit:** 2:0:0
* **Reservation:** LSC-ML-Institute-May19

.. warning::

   The reservation name changes day by day.

.. image::  ./images/TAP_1.job_submittting.png
   :alt:  Figure 1. Submitting a job through TAP 

* Click 'Submit' and wait for the job to start
* Click 'Connect' when the a node becomes available

.. image::  ./images/TAP_2.job_connect.png
   :alt:  Figure 2 Submitting a job through TAP 
 
* The TAP job will open the user ``$HOME`` directory. Open ``Mushroom-ANN-tutorial.ipynb`` or
  ``Coral-CNN-tutorial.ipynb``.

.. image::  ./images/TAP_3.jupyter_HOME.png
   :alt:  Figure 3 TAP session will log into user $HOME 

* Change your kernel to ``Day3-tf-213``. Click on the menu ``kernel``, then ``Change kernel``, and select the kernel ``Day3-tf-213``. Trust the kernel by clicking on the button "Not trusted" at the top right 

.. image::  ./images/TAP_4.kernel_change.png
   :alt:  Figure 4 Changing the kernel version ant trust the kernel

* The Jupyter notebook will ask confirmation before trusting the kernel.

.. image::  ./images/TAP_5.jupyter.trusting.png
   :alt:  Figure 5 Kernel trusting confirmation

* After clicking "trust" on the confirmation button, the button at the top right will appear as "Trusted".

.. image::  ./images/TAP_6.jupyter.trusted.png
   :alt:  Figure 6 Kernel trusted

* The Jupyer notebook will be ready to be run. Note: The kernel may take a few moments to initialize on first use.


Complete the Tutorial
^^^^^^^^^^^^^^^^^^^^^

To complete this tutorial:

1. Follow the step-by-step instructions on our
   `ReadTheDocs <https://life-sciences-ml-at-tacc.readthedocs.io/en/latest/section3/overview.html>`_.
2. Write the code from the ReadTheDocs page into the corresponding empty cells in your notebook.
3. Execute each cell to build your ANN/CNN and see the results.

If you get stuck, a completed solution is available within the ``tutorials`` directory of the
repository you cloned previously.


Check GPU Availability
^^^^^^^^^^^^^^^^^^^^^^

Before training deep learning models on HPC systems, it's important to check whether TensorFlow can
access the GPU. Training on a GPU is significantly faster than on a CPU, especially for large image
datasets.

If you've followed the setup instructions in the previous section, and you've run the
``install_kernel.sh`` script on Frontera, you should now be running the tutorial notebook inside a
containerized Jupyter kernel that includes:

* TensorFlow (v2.13) with GPU support
* CUDA libraries compatible with the system 
* All required Python packages pre-installed

To confirm that your environment is correctly configured, run the following code cell in the
tutorial notebook:

.. tip::

   Make sure to change your kernel to ``Day3-tf-213``.

.. code-block:: python

   >>> import tensorflow as tf
   
   >>> # Check if TensorFlow can detect the GPU
   >>> print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
   
   >>> # Print TensorFlow version
   >>> print(tf.__version__)


Set Up For Vista
---------------- 

Step 1. Log in to Vista
^^^^^^^^^^^^^^^^^^^^^^^

Log in to Vista using SSH:

.. code:: console

   [local]$ ssh username@vista.tacc.utexas.edu
   (username@vista.tacc.utexas.edu) Password: 
   (username@vista.tacc.utexas.edu) TACC Token Code:

   # ------------------------------------------------------------------------------
   # Welcome to the Vista Supercomputer
   # Texas Advanced Computing Center, The University of Texas at Austin
   # ------------------------------------------------------------------------------

Step 2. Set up Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Download the setup script (``install_kernels.sh``) and run it to set up the environment. This script will copy the Jupyter
kernel image files into your SCRATCH directory and install the kernel definition files into your HOME directory.

.. code:: console

   # Change to your SCRATCH directory
   [vista]$ cds
   
   # Download the setup script
   [vista]$ wget https://raw.githubusercontent.com/TACC/life_sciences_ml_at_tacc/refs/heads/main/docs/section4/files/install_kernels.sh

   # Run the setup script
   [vista]$ bash ./install_kernels.sh
   Copying tensorflow-ml-container_0.1.sif to /scratch/03762/eriksf...
   Copying pytorch-ml-container_0.4.sif to /scratch/03762/eriksf...
   Kernel directory created at ~/.local/share/jupyter/kernels/Day4-tf-217 and kernel.json has been added.
   Kernel directory created at ~/.local/share/jupyter/kernels/Day4-pt-251 and kernel.json has been added.

Step 3. Launch Jupyter
^^^^^^^^^^^^^^^^^^^^^^

Log in to the `TACC Analysis Portal <https://tap.tacc.utexas.edu/jobs/>`_ and configure your session
as follows:

* **System:** Vista
* **Application:** Jupyter Notebook
* **Project:** frontera-training
* **Queue:** gh
* **Reservation:** LSC-ML-Institute-May22

.. warning::

   The reservation name changes day by day.

.. image::  ./images/TAP1_vista_job_submitting.png
   :alt:  Figure 1. Submitting a job through TAP 

* Click 'Submit' and wait for the job to start
* Click 'Connect' when the a node becomes available

.. image::  ./images/TAP2_vista_connect.png
   :alt:  Figure 2 Submitting a job through TAP 
 
* By default on Vista, the Jupyter Notebook job will open with the Jupyter Lab interface showing the user
  ``$HOME`` directory on the left. If the kernels are installed properly, you should see the
  ``Day4-tf-217`` and ``Day4-pt-251`` kernels listed in the Launcher tab under the Notebook section.

.. image::  ./images/TAP3_vista_jupyter_lab_home.png
   :alt:  Figure 3 Jupyter Lab interface showing user $HOME 

.. note::

   If you prefer to use the classic Jupyter Notebook interface instead of Jupyter Lab, you can edit the URL
   in your browser to replace the word "/lab" with "/tree". 

* To verify that the kernels are installed properly in the Jupyter Notebook interface, click on the "New" dropdown
  menu in the upper right to see the ``Day4-tf-217`` or ``Day4-pt-251`` kernels.

.. image::  ./images/TAP4_vista_jupyter_notebook_home.png
   :alt:  Figure 4 Jupyter Notebook interface showing user $HOME

* The Jupyter notebooks are now ready to be launched.

.. note::

   The kernel may take a few moments to initialize on first use.
