Executing Basic Job Management Tasks
====================================

First, we must know an application we want to run, and a research question we want to ask. 
In this example, we aim to execute a Python code that determines the larger of two numbers. 
The code also includes a 3-second delay before finishing.

.. code-block:: console

   [frontera]$ cdw
   [frontera]$ cd Lab01
   [frontera]$ pwd
   /work2/02555/lima/frontera/Lab01
   [frontera]$ ls
   example.slurm  example_template.slurm  my_code.py

Next, we need to fill out ``example_template.slurm`` to request the necessary resources. 
I know that this code will take a little more than 3 seconds, so I can reasonably predict how much we will need. 
When running your first jobs with your applications, it will take some trial and error, and reading online documentation, 
to get a feel for how many resources you should use. Open ``example_template.slurm`` with VIM and fill out the following information:

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

.. code-block:: bash
 
   [frontera]$ vim example_template.slurm

There are two "modes" in VIM that we will talk about today. They are called "insert mode" and "normal mode". 
In insert mode, the user is typing text into a file as seen through the terminal (think about typing text into TextEdit or Notepad). 
In normal mode, the user can perform other functions like save, quit, cut and paste, find and replace, etc. 
(think about clicking the menu options in TextEdit or Notepad). The two main keys to remember to toggle between the modes are ``i`` and ``Esc``.

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

Once you have filled in the job description, save and quit the file. 
Submit the job to the queue using the ``sbatch`` command`:

.. code-block:: console

   $ sbatch example_template.slurm

To view the jobs you have currently in the queue, use the ``showq`` or ``squeue`` commands:

.. code-block:: console

   $ showq -u
   $ showq        # shows all jobs by all users
   $ squeue -u $USERNAME
   $ squeue       # shows all jobs by all users

If for any reason you need to cancel a job, use the ``scancel`` command with the 6- or 7-digit jobid:

.. code-block:: console

   $ scancel jobid

For more example scripts, see this directory on Frontera:

.. code-block:: console

   $ ls /share/doc/slurm/

If everything went well, you should have a file named ``duration.txt``, 
an output file named something similar to ``output.o6146935``, 
and an error file named something similar to ``error.o6146935`` in the same directory as the ``example_template.slurm`` script. 

.. code-block:: console

   [frontera] $ ls
   duration.txt  error.6146935  example.slurm  example_template.slurm  my_code.py  output.6146935

**Congratulations! You ran a batch job on Frontera!**

Review of VIM Commands Covered
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

+------------------------------------+-------------------------------------------------+
| Command                            |          Effect                                 |
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
