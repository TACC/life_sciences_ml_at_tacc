Transferring Files
==================

Creating and Changing Folders
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On a Windows or Mac desktop, our present location determines what files and folders
we can access. I can "see" my present location visually with the help of the graphic
interface - I could be looking at my Desktop, or the contents of a folder, for example.
In a Linux command-line interface, we lack the same visual cues to tell us what our
location is. Instead, we use a command - ``pwd`` (print working directory) - to tell
us our present location. Try executing this command on Frontera:

.. code-block:: console

   [frontera]$ pwd
   /home1/02555/lima

This home location on the Linux filesystem is unique for each user, and it is roughly
analogous to C:\\Users\\username on Windows, or /Users/username on Mac.

To see what files and folders are available at this location, use the ``ls`` (list) command:

.. code-block:: console

   [frontera]$ ls

I have no files or folders in my home directory yet, so I do not get a response.
We can create some folders using the ``mkdir`` (make directory) command. The words 
'folder' and 'directory' are interchangeable:

.. code-block:: console

   [frontera]$ mkdir folder1
   [frontera]$ mkdir folder2

.. code-block:: console

   [frontera]$ ls
   folder1 folder2

Now we have some folders to work with. To "open" a folder, navigate into that folder 
using the ``cd`` (change directory) command. This process is analogous to double-clicking 
a folder on Windows or Mac:

.. code-block:: console

   [frontera]$ pwd
   /home1/02555/lima
   [frontera]$ cd folder1
   [frontera]$ pwd
   /home1/02555/lima/folder1

Use ``ls`` to list the contents. What do you expect to see?

.. code-block:: console

   [frontera]$ ls

There is nothing there because we have not made anything yet. Next, we will navigate back to the 
home directory. So far we have seen how to navigate "down" into folders, but how do we navigate 
back "up" to the parent folder? There are different ways to do it. For example, we could use a shortcut, ``..``, 
which refers to the **parent folder** - one level higher than the current location:

.. code-block:: console

   [frontera]$ cd ..
   [frontera]$ pwd
   /home1/02555/lima

We are back in our home directory. Instead, we could specify 
the complete path of where we want to go, in this case ``cd /home1/02555/lima``.
Finally, let's remove the directories we have made, using ``rm -r`` to remove our parent 
folder ``folder1`` and its subfolders. The ``-r`` command line option recursively removes subfolders 
and files located "down" the parent directory. ``-r`` is required for folders.

.. code-block:: console

   [frontera]$ rm -r folder1
   [frontera]$ ls 
   folder2


Transferring Files to and from Frontera
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To practice transferring files to Frontera's ``$WORK`` and ``$SCRATCH``, we need to identify the path to our ``$WORK`` and ``$SCRATCH`` directory. 
To identify these paths, we can use helpful command shortcuts.

To identify the path to our ``$WORK`` directory, we can use ``cd $WORK`` or the helpful shortcut ``cdw``:

.. code-block:: console
   
   [frontera]$ cdw
   [frontera]$ pwd
   /work2/02555/lima/frontera

To identify the path to our ``$SCRATCH`` directory, we can use ``cd $SCRATCH`` or the helpful shortcut ``cds``:

.. code-block:: console
   
   [frontera]$ cds
   [frontera]$ pwd
   /scratch1/02555/lima/frontera

Copying files from your local computer to Frontera's ``$WORK`` would require the ``scp`` command (Windows users use the program "WinSCP"):

.. code-block:: console

   [local]$ scp my_file lima@frontera.tacc.utexas.edu:/work2/02555/lima/frontera
   (enter password)
   (enter token)

In this command, you specify the name of the file you want to transfer (``my_file``), the username (``lima``), the hostname (``frontera.tacc.utexas.edu``), 
and the path you want to put the file (``/work2/02555/lima/frontera``). Take careful notice of the separators including spaces, the @ symbol, and the colon. 

Copying files from your local computer to Frontera's ``$SCRATCH`` using ``scp``:

.. code-block:: console

   [local]$ scp my_file lima@frontera.tacc.utexas.edu:/scratch1/02555/lima/frontera
   (enter password)
   (enter token)

Copy files from Frontera to your local computer using the following:

.. code-block:: console

   [local]$ scp lima@frontera.tacc.utexas.edu:/work2/02555/lima/frontera/my_file ./
   (enter password)
   (enter token)

Note: If you wanted to copy ``my_file`` from ``$SCRATCH``, the path you would specify after the colon would be ``/scratch1/02555/lima/frontera/my_file``.
 
Instead of files, full directories can be copied using the "recursive" flag (``scp -r ...``). 

This is just the basics of copying files. See example ``scp`` usage `here <https://en.wikipedia.org/wiki/Secure_copy>`_.

Exercise
^^^^^^^^

1. Download the `file firststeps.zip <https://github.com/Ernesto-Lima/YourFirstStepsAtTACC/raw/master/docs/steps_at_tacc/files/firststeps.zip>`_.

2. Login to Frontera.

3. Identify your ``$WORK`` directory path using ``cdw`` and ``pwd``.

4. From your local computer, copy the file ``firststeps.zip`` to Frontera. (You will need to know where the file ``firststeps.zip`` was downloaded on your local computer and navigate to this folder.)

5. Login to Frontera, navigate to your ``$WORK``, and unzip the file using ``unzip firststeps.zip``.

.. toggle:: Click to show the answer

   1. Download the `file firststeps.zip <https://github.com/Ernesto-Lima/YourFirstStepsAtTACC/raw/master/docs/steps_at_tacc/files/firststeps.zip>`_.

   2. Login to Frontera:

      .. code-block:: console
   
         [local]$ ssh username@frontera.tacc.utexas.edu
         (enter password)
         (enter 6-digit token)
   
   3. Identify your ``$WORK`` directory path using ``cdw`` and ``pwd``.

      .. code-block:: console
   
         [frontera]$ cdw
         [frontera]$ pwd
         /work2/02555/lima/frontera
         [frontera]$ logout

   4. From your local computer, copy the file ``firststeps.zip`` to Frontera. (You will need to know where the file ``firststeps.zip`` was downloaded on your local computer and navigate to this folder.)

      .. code-block:: console

         [local]$ scp firststeps.zip lima@frontera.tacc.utexas.edu:/work2/02555/lima/frontera
         (enter password)
         (enter token)

   5. Login to Frontera, navigate to your ``$WORK``, and unzip the file using ``unzip firststeps.zip``.

      .. code-block:: console
         
         [local]$ ssh username@frontera.tacc.utexas.edu
         (enter password)
         (enter 6-digit token)
         [frontera]$ cdw
         [frontera]$ unzip firststeps.zip



Review of Topics Covered
^^^^^^^^^^^^^^^^^^^^^^^^

+------------------------------------+-------------------------------------------------+
| Command                            |          Effect                                 |
+====================================+=================================================+
| ``pwd``                            |  print working directory                        |
+------------------------------------+-------------------------------------------------+
| ``ls``                             |  list files and directories                     |
+------------------------------------+-------------------------------------------------+
| ``mkdir dir_name``                 |  make a new directory                           |
+------------------------------------+-------------------------------------------------+
| ``cd dir_name/``                   |  navigate into a directory                      |
+------------------------------------+-------------------------------------------------+
| ``rm -r dir_name/``                |  remove a directory and its contents            |
+------------------------------------+-------------------------------------------------+
| ``.`` or ``./``                    |  refers to the present location                 |
+------------------------------------+-------------------------------------------------+
| ``..`` or ``../``                  |  refers to the parent directory                 |
+------------------------------------+-------------------------------------------------+
| ``cd $WORK``, ``cdw``              |  Navigate to ``$WORK`` file system              |
+------------------------------------+-------------------------------------------------+
| ``cd $SCRATCH``, ``cds``           |  Navigate to ``$SCRATCH`` file system           |
+------------------------------------+-------------------------------------------------+
| ``scp local remote``               |  Copy a file from local to remote               |
+------------------------------------+-------------------------------------------------+
| ``scp remote local``               |  Copy a file from remote to local               |
+------------------------------------+-------------------------------------------------+