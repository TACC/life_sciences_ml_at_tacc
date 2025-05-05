TACC Analysis Portal
====================

Over the next several units we will use the `TACC Analysis Portal <https://tap.tacc.utexas.edu/>`_
(TAP) to run interactive Jupyter Notebook sessions on TACC systems. TAP targets users who want the
convenience of web-based portal access while maintaining control over low-level job behavior.  

Any user with an allocation on one of TACC's HPC Systems, e.g. Frontera, Vista, Stampede3, and
Lonestar6, has access to the TACC Analysis Portal. TAP-Supported applications include:

* DCV (Desktop Cloud Visualization) remote desktop
* VNC (Virtual Network Computing) remote desktop
* **Jupyter Notebook**
* RStudio

By the end of this section, you should be able to:

* Log in to the TACC Analysis Portal
* Submit a Jupyter Notebook job to a TACC system
* Connect to a Jupyter Notebook session
* Choose the correct Jupyter kernel
* Run simple Python code in a Jupyter Notebook




Accessing the Portal
--------------------

Log in to TACC Analysis Portal using the same username and password that you use to access the TACC
User Portal. Once you've logged in you'll be directed to the Home Screen where you can begin
scheduling jobs.

.. image::  ./images/1TAP.png
   :target: ./images/1TAP.png
   :alt:  Figure 1. TAP Home Screen


Submitting a Job
~~~~~~~~~~~~~~~~

.. raw:: html

 <span style="text-align: justify; font-size: 16px;line-height:24px;">Submitting a job on TAP requires the following inputs:</span>  
 <span style="background-color:#FF7F00;color:white;"><b>&nbsp( 1 )&nbsp</b></span>
..
 .. raw:: html

    <style> .red {color:#f09837; font-weight:bold; font-size:16px} </style>
    <span style="background-color:#f2a024;color:white;">( 1 )</span>

* **System:** where the job will run. The system selector drop-down will contain the TAP-supported TACC systems where you have an allocation. The system must be selected first, as the values of the other selectors are determined by the selected system. 
* **Application:** which application the job will run. The application selector will contain the applications available on the selected system (DCV, VNC,Jupyter, or RStudio)
* **Project:** which project allocation to bill for the job run. The project selector will contain the projects associated with your account on the selected system.  
* **Queue:** which system queue will receive the job. The queue selector will contain the TAP-supported queues on the selected system.  
* **Nodes:** the number of nodes the job will occupy. We recommend leaving this setting at 1 unless you know you need more nodes. This is equivalent to the `-N` option in SLURM.  
* **Tasks:** the number of MPI tasks the job will use. We recommend leaving this setting at 1 unless you know you need more tasks. This is equivalent to the `-n` option in SLURM.  

..  <span style="background-color:#FF7F00; color:#FFFFFF;">(&nbsp;2&nbsp;)</span>
 
.. raw:: html

 <span style="text-align: justify; font-size: 16px;line-height:24px;">A TAP job also accepts these additional optional inputs:</span>  
 <span style="background-color:#FF7F00;color:white;"><b>&nbsp( 2 )&nbsp</b></span>
 
* **Time Limit:** how long the job will run. If left blank, the job will use the TAP default runtime of 2 hours.  
* **Reservation:** the reservation in which to run the job. If you have a reservation on the selected system and want the job to run within that reservation, specify the name here.  
* **VNC Desktop Resolution:** desktop resolution for a VNC job. If this is left blank, a VNC job will use the default resolution of 1024x768.  


.. raw:: html

 <span style="text-align: justify; font-size: 16px;line-height:24px;">After specifying the job inputs, select the <b>Submit</b> </span>
 <span style="background-color:#FF7F00;color:white;"><b>&nbsp( 8 )&nbsp</b></span>   
 <span style="text-align: justify; font-size: 16px;line-height:24px;"> button, and your job will be submitted to the remote system. After submitting the job, you will be automatically redirected to the job status page. You can get back to this page from the <b>Status</b> 
 <span style="background-color:#FF7F00; color:#FFFFFF;"><b>&nbsp( 3 )&nbsp</b></span> 
 <span style="text-align:justify;font-size: 16px;line-height:24px;"> button. If the job is already running on the system, click the</span><b> Connect</b> 
 <span style="background-color:#FF7F00; color:#FFFFFF;"><b>&nbsp( 5 )&nbsp</b> </span>
 <span style="text-align:justify;font-size: 16px;line-height:24px;"> button from the Home Screen or Job status to connect to your application.</span>

|

.. image::  ./images/2TAP.png
   :target: ./images/2TAP.png
   :alt:  Figure 2. Job Status

|

Click the "Check Status" button to update the page with the latest job status. The diagnostic information will include an estimated start time for the job if Slurm is able to provide one. Jobs submitted to development queues typically start running more quickly than jobs submitted to other queues.

Ending a Submitted Job 
~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

 <span style="text-align: justify;font-size: 16px;line-height:24px;">When you are finished with your job, you can end your job using the </span><b>End</b> 
 <span style="background-color:#FF7F00;color:white;"><b>&nbsp( 4 )&nbsp</b></span>   
 <span style="text-align: justify; font-size: 16px;line-height:24px;">button on the TAP Home Screen page or on the Job Status page. Note that closing the browser window will not end the job. Also note that if you end the job from within the application (for example, pressing "Enter" in the red xterm in a DCV or VNC job), TAP will still show the job as running until you check status for the job, click "End Job" within TAP, or the requested end time of the job is reached.</span>


Resubmitting a Past Job
~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <span style="text-align: justify;font-size: 16px;line-height:24px;">You can resubmit a past job using the</span><b> Resubmit </b>
    <span style="background-color:#FF7F00;color:white;"><b>&nbsp( 7 )&nbsp</b></span> 
    <span style="text-align: left;font-size: 16px;line-height:24px;">button from the Home Screen page. The job will be submitted with the same inputs used for the past job, including any optional inputs. Select </span> <b>Details</b> 
    <span style="background-color:#FF7F00; color:#FFFFFF;"><b>&nbsp( 6 ) </b></span>&nbsp; 
    <span style="text-align: justify;font-size: 16px;line-height:24px;">to see the inputs that were specified for the past job.</span> 

|

.. image::  ./images/3TAP.png
   :target: ./images/3TAP.png
   :width: 300
   :align: center
   :alt:  Figure 3. TAP Job Details

|
 

Utilities
~~~~~~~~~

.. raw:: html

    <span style="text-align: justify;font-size: 16px;line-height:24px;">TAP provides certain useful diagnostic and logistic utilities on the Utilities page. Access the Utilities page by selecting the <b>Utilities</b> <span    style="background-color:#FF7F00; color:#FFFFFF;"><b>&nbsp( 9 ) </b></span> &nbsp;button on the Home Screen page. 


.. image::  ./images/4TAP.png
   :target: ./images/3TAP.png
   :align: center
   :alt:  e 4. TAP Utilities



Configuring Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Utilities section provides access to several common actions related to Jupyter Notebooks. 
* **"Use Python3"** sets the TACC Python3 module as part of your default modules so that TAP will use Python3 for Jupyter Notebooks. If you want to use a non-default Python installation, such as Conda, you will need to install it yourself via the system command line. TAP will use the first "jupyter-notebook" command in your `$PATH`, so make sure that the command "which jupyter-notebook" returns the Jupyter Notebook you want to use. Conda install typically configures your environment so that Conda is first on your `$PATH`.

"Link `$WORK` from `$HOME`" and "Link `$SCRATCH` from `$HOME`" create symbolic links in your `$HOME` directory so that you can access `$WORK` and `$SCRATCH` from within a Jupyter Notebook. TAP launches Jupyter Notebooks from within your `$HOME` directory, so these other file systems are not reachable without such a linking mechanism. The links will show up as "WORK" and "SCRATCH" in the Jupyter file browser. You only need to create these links once and they will remain available for all future jobs.

Obtaining TACC Account Status 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Status section provides system information and diagnostics. "Check TACC Info" will show account balances and filesystem usage for the selected system. "Run Sanity Tool" performs a series of sanity checks to catch common account issues that can impact TAP jobs (for example, being over filesystem quota on your `$HOME` directory).

Setting a Remote Desktop to Full Screen Mode  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both DCV and VNC support full-screen mode. DCV will automatically adjust the desktop resolution to use the full screen, whereas VNC will keep the original desktop resolution within the full-screen view.

In DCV, click the Fullscreen button in the upper left corner of the DCV desktop.

.. image::  ./images/5TAP.png
   :target: ./images/5TAP.png
   :align: center
   :alt:  Figure 5. DCV Full Screen
.. 
  #   :width: 300

|


In VNC, open the control bar on the left side of the screen, then click the Fullscreen button.

.. image::  ./images/6TAP.png
   :target: ./images/6TAP.png
   :align: center
   :alt:  Figure 6. VNC Full Screen

|
 

Troubleshooting 
~~~~~~~~~~~~~~~

* **No Allocation Available** If TAP cannot find an allocation for your account on a supported system, you will see the message below. If the issue persists, [create a ticket][HELPDESK] in the TACC Consulting System.

.. image::  ./images/7TAP.png
   :target: ./images/7TAP.png
   :align: center
   :alt:  Figure 7. TAP Error: No Allocation

* **Job Submission returns PENDING** If the job does not start immediately, TAP will load a status page with some diagnostic information. If the job status is "PENDING", the job was successfully submitted and has not yet started running. If Slurm can predict when the job will start, that information will be in the `squeue --start` output in the message window. Clicking the "Check Status" button will update the job status. When the job has started, TAP will show a "Connect" button.

.. image::  ./images/8TAP.png
   :target: ./images/8TAP.png
   :align: center
   :alt:  Figure 8. TAP Error: PENDING


* **Job Submission returns ERROR** If the TAP status page shows that the job status is "ERROR", then there was an issue with the Slurm submission, and the message box will contain details. If you have difficulty interpreting the error message or resolving the issue, please create a ticket in the TACC Consulting System and include the TAP message.
 
.. image::  ./images/9TAP.png
   :target: ./images/9TAP.png
   :align: center
   :alt:  Figure 9. TAP "Error"




Set up for Deep Learning Tutorial
---------------------------------

This repository contains hands-on tutorials and materials that accompany
the `Deep Learning
section <https://life-sciences-ml-at-tacc.readthedocs.io/en/latest/section3/overview.html>`__
of the Life Sciences Machine Learning Institute at the `Texas Advanced
Computing Center (TACC) <https://tacc.utexas.edu/>`__.

.. _1-accessing-frontera:

1. Accessing Frontera
~~~~~~~~~~~~~~~~~~~~~

Log into Frontera using SSH:

.. code:: bash

   ssh username@frontera.tacc.utexas.edu
   (username@frontera.tacc.utexas.edu) Password: 
   (username@frontera.tacc.utexas.edu) TACC Token Code:

   # ------------------------------------------------------------------------------
   # Welcome to the Frontera Supercomputer
   # Texas Advanced Computing Center, The University of Texas at Austin
   # ------------------------------------------------------------------------------

.. _2-getting-the-tutorial-materials:

2. Getting the Tutorial Materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Navigate to your scratch directory and clone this tutorial repository:

.. code:: bash

   cds # shortcut for cd #SCRATCH
   git clone https://github.com/kbeavers/tacc-deep-learning-tutorials.git

.. _3-environment-setup:

3. Environment Setup
~~~~~~~~~~~~~~~~~~~~

.. _a-start-an-interactive-session:

a. Start an Interactive Session
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   cds
   idev -m 20

.. _b-set-up-the-container-environment:

b. Set up the Container Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   # Load the Apptainer module
   module load tacc-apptainer

   # Pull the Docker container image created for this tutorial
   apptainer pull docker://kbeavers/tf-213:frontera

   # Run the kernel setup script
   cd tacc-deep-learning-tutorials
   bash ./scripts/install_kernel.sh

.. _4-dataset-preparation:

4. Dataset Preparation
~~~~~~~~~~~~~~~~~~~~~~

Extract the provided coral species image dataset

.. code:: bash

   bash ./scripts/download_dataset.sh

.. _5-launching-the-tutorial:

5. Launching the Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _a-copy-the-tutorial-notebooks-to-your-home-directory:

a. Copy the tutorial notebooks to your home directory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   cp ./tutorials/Mushroom-ANN-tutorial.ipynb $HOME/
   cp ./tutorials/Coral-CNN-tutorial.ipynb $HOME/

These notebooks are provided as blank templates for you to fill in as
you work through the exercises. To complete this tutorial:

1. Follow the step-by-step instructions on our
   `ReadTheDocs <https://life-sciences-ml-at-tacc.readthedocs.io/en/latest/section3/overview.html>`__.
2. Write the code from the ReadTheDocs page into the corresponding empty
   cells in your notebook.
3. Execute each cell to build your ANN/CNN and see the results.

If you get stuck, a completed solution is available within the
``tutorials`` directory of this repository.

.. _b-access-the-tacc-analysis-portal-and-configure-your-session-as-follows:

b. Access the `TACC Analysis Portal <https://tap.tacc.utexas.edu/jobs/>`__ and configure your session as follows:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  System: Frontera
-  Application: Jupyter Notebook
-  Project:
-  Queue: rtx
-  Job Name: DL-Training
-  Time Limit: 2:0:0
-  Reservation: (or leave blank if no reservation)

.. _c-final-steps:


.. image::  ./images/TAP_1.job_submittting.png
   :target: ./images/TAP_1.job_submittting.png
   :alt:  Figure 1. Submitting a job through TAP 



c. Final Steps:
^^^^^^^^^^^^^^^

-  Click 'Submit' and wait for the job to start
-  Click 'Connect' when the a node becomes available

.. image::  ./images/TAP_2.job_connect.png
   :target: ./images/TAP_2.job_connect.png
   :alt:  Figure 2 Submitting a job through TAP 
 

- The TAP job will open the user $HOME directory. Open ``Mushroom-ANN-tutorial.ipynb`` or ``Coral-CNN-tutorial.ipynb``.

.. image::  ./images/TAP_3.jupyter_HOME.png
   :target: ./images/TAP_3.jupyter_HOME.png
   :alt:  Figure 3 TAP session will log into user $HOME 


-  Change your kernel to ``tf-213``. Click on the menu ``kernel``, then ``Change kernel``, and select the kernel ``tf-213``. Trust the kernel by clicking on the button "Not trusted" at the top right 

.. image::  ./images/TAP_4.kernel_change.png
   :target: ./images/TAP_4.kernel_change.png
   :alt:  Figure 4 Changing the kernel version ant trust the kernel

-  The Jupyter notebook will ask confirmation before trusting the kernel.

.. image::  ./images/TAP_5.jupyter.trusting.png
   :target: ./images/TAP_5.jupyter.trusting.png
   :alt:  Figure 5 Kernel trusting confirmation

-  After clicking "trust" on the confirmation button, the button at the top right will appear as "Trusted".

.. image::  ./images/TAP_6.jupyter.trusted.png
   :target: ./images/TAP_6.jupyter.trusted.png
   :alt:  Figure 6 Kernel trusted

-  The Jupyer notebook will be ready to be run. Note: The kernel may take a few moments to initialize on first use.







Set up for the CNN Tutorial
---------------------------

On Day 3 we will run a hands-on Convolutional Neural Network (CNN) tutorial.
Here we provide the instruction to retrieve the necessary files and set up the enviroment.

1. Accessing Frontera
~~~~~~~~~~~~~~~~~~~~~

Log in to Frontera using SSH:

.. code-block:: bash
    
    ssh username@frontera.tacc.utexas.edu
    (username@frontera.tacc.utexas.edu) Password: 
    (username@frontera.tacc.utexas.edu) TACC Token Code:

    # ------------------------------------------------------------------------------
    # Welcome to the Frontera Supercomputer
    # Texas Advanced Computing Center, The University of Texas at Austin
    # ------------------------------------------------------------------------------

2. Getting the Tutorial Materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Navigate to your $SCRATCH directory and clone the tutorial repository:

.. code-block:: bash

    cds  # shortcut for cd $SCRATCH
    git clone git@github.com:kbeavers/coral-species-CNN-tutorial.git

3. Environment Setup
~~~~~~~~~~~~~~~~~~~~

a) Start an Interactive Session:

.. code-block:: bash

    cds
    idev -m 20

b) Set up the Container Environment:

.. code-block:: bash

    # Load the apptainer module
    module load tacc-apptainer

    # Pull the Docker container image created for this tutorial
    apptainer pull docker://kbeavers/tf-cuda101-frontera:0.1

    # Run the kernel setup script
    bash ./coral-species-CNN-tutorial/scripts/install_kernel.sh

4. Dataset Preparation
~~~~~~~~~~~~~~~~~~~~~~

Extract the provided coral species image dataset:

.. code-block:: bash

    cd coral-species-CNN-tutorial
    bash ./scripts/download_dataset.sh

5. Launching the Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~

a) Copy the tutorial notebook to your $HOME directory:

.. code-block:: bash

    cp ./tutorials/Coral-CNN.ipynb $HOME/

b) Access the TACC Analysis Portal and configure your session as follows:

   - System: Frontera
   - Application: Jupyter Notebook
   - Project: <your-allocation>
   - Queue: rtx
   - Job Name: CNN-Training
   - Time Limit: 2:0:0
   - Reservation: <your-reservation>

c) Final Steps:

   - Click 'Submit' and wait for the job to start
   - Click 'Connect' when available
   - Open ``Coral-CNN.ipynb`` in your $HOME directory
   - Change your kernel to ``tf-cuda101``
   - Trust the kernel if necessary

Note: The kernel may take a few moments to initialize on first use. 



6. Check GPU Availability
~~~~~~~~~~~~~~~~~~~~~~~~~

Before training deep learning models on HPC systems, it's important to check whether TensorFlow can access the GPU. 
Training on a GPU is significantly faster than on a CPU, especially for large image datasets.

If you've followed the setup instructions in the previous section, and you've run the ``install_kernel.sh`` script on Frontera, you should now be running the tutorial notebook inside a containerized Jupyter kernel that includes:

- TensorFlow (v. _____) with GPU support
- CUDA libraries compatible with the system 
- All required Python packages pre-installed

To confirm that your environment is correctly configured, run the following code cell in the tutorial notebook (TIP: Make sure to change your kernel to ``tf-cuda101``):

.. code-block:: python

    import tensorflow as tf

    # Check if TensorFlow can detect the GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Print TensorFlow version
    print(tf.__version__)









