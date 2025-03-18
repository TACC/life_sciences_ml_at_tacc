Running Interactive Sessions with Idev
======================================

The ``idev`` utility initiates an interactive session on one or more compute nodes
so that you can issue commands and run code as if you were doing so on your personal
machine. An interactive session is useful for development, as you can quickly compile,
run, and validate code. Accessing a single compute node is accomplished by simply
executing ``idev`` on any of the TACC systems.

Initiating an Interactive Session
---------------------------------

To learn about the command line options available for ``idev``, use ``idev -help``.

.. code-block:: console
   
   login1$ idev -help
   ...
   OPTION ARGUMENTS         DESCRIPTION
   -A     account_name      sets account name (default: in .idevrc)
   -m     minutes           sets time in minutes (default: 30)
   -p     queue_name        sets queue to named queue (default: -p development)
   -r     resource_name     sets hardware
   -t     hh:mm:ss          sets time to hh:mm:ss (default: 00:30:00)
   -help      [--help     ] displays (this) help message
   -v         [--version  ] output version information and exit

Let's go over some of the most useful ``idev`` command line options that can customize your interactive session:

To change the **time** limit to be lesser or greater than the default 30 minutes, users can use the ``-m`` command line option. 
For example, a user requesting an interactive session for an hour would use the command line option ``-m 60``.

To change the **account_name** associated with the interactive session, users can use the ``-A`` command line option. 
This option is useful for when a user has multiple allocations they belong to. 
For example, if I have allocations on accounts ``TACC`` and ``Training``, 
I can use ``-A`` to set the allocation I want to be used like so: ``-A TACC`` or ``-A Training``.

To change the **queue** to be different than the default ``development`` queue, users can use the ``-p`` command line option. 
For example, if a user wants to launch an interactive session on one of Frontera GPU nodes, 
they would use the command line option ``-p rtx`` or ``-p rtx-dev``. 
You can learn more about the different queues of Frontera `here <https://docs.tacc.utexas.edu/hpc/frontera/#table6>`_.

Note: For the scope of this section, we will be using the default ``development`` queue.  

To start a thirty-minute interactive session on a compute node in the development queue with our ``EXAMPLE`` allocation:

.. code-block:: console
   
   login1$ idev -A EXAMPLE   

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

Exercise
--------

Let's execute a Python code that determines the larger of two numbers. 
The code also includes a 3-second delay before finishing.

.. code-block:: console

   c205-004[clx](634)$ cdw
   c205-004[clx](635)$ unzip firststeps.zip
   c205-004[clx](636)$ cd Lab01
   c205-004[clx](637)$ pwd
   /work2/02555/lima/frontera/Lab01
   c205-004[clx](638)$ ls
   example.slurm  example_template.slurm  my_code.py

Load the appropriate modules, and run ``my_code.py``. 

.. code-block:: console

   c205-004[clx](639)$ module load python3
   c205-004[clx](640)$ python3 my_code.py
   The larger number of 51 and 20 is 51

You can check the files that were generated using ``ls``, and see the contents of the file with ``cat``.

.. code-block:: console

   c205-004[clx](641)$ ls
   duration.txt  example.slurm  example_template.slurm  my_code.py
   c205-004[clx](642)$ cat duration
   Done in 3.009739637374878 seconds.

To exit an interactive session, you can use the command ``logout``.

Attention
^^^^^^^^^

In case you didn't donwload the code in the section :doc:`Transferring Files<transfering>`, you can download by doing:

.. code-block:: console

   [frontera]$ cdw
   [frontera]$ wget https://github.com/Ernesto-Lima/YourFirstStepsAtTACC/raw/master/docs/steps_at_tacc/files/firststeps.zip
   [frontera]$ unzip firststeps.zip
