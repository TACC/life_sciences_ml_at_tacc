Managing Modules
================

Modules enable users to run specific applications or access libraries without the need to log out and back in. 
Modules for applications adjust the user's path for easy access, while those for library packages set environment 
variables indicating the location of library and header files. 
Switching between package versions or removing a package is straightforward.

User's Tour of the Module Command
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Press the ``<Enter>`` key to scroll through line-by-line, or the ``<Space>`` key to scroll through page-by-page. 
Press ``q`` to quit the view.

If there are many modules on a system, it can be difficult to see what
modules are available to load. To display a concise listing:

.. code-block:: console

    [frontera]$ module overview

    ------------------ /opt/apps/modulefiles/Core -----------------
    StdEnv    (1)   hashrf    (2)   papi        (2)   xalt     (1)
    ddt       (1)   intel     (2)   singularity (2)   
    git       (1)   noweb     (1)   Rstats      (2)

    --------------- /opt/apps/lmod/lmod/modulefiles/Core ----------
    lmod (1)   settarg (1)

This shows the short name of the module (i.e. git, or singularity)
and the number in the parenthesis is the number of versions for each.
This list above shows that there is one version of git and two
versions of Rstats.

To check all the versions of a package (e.g., Rstats):

.. code-block:: console

    [frontera]$ module avail Rstats

    --------------- /opt/apps/intel19/impi19_0/modulefiles ---------------
         Rstats/3.6.3    Rstats/4.0.3 (D)

In *Rstats/4.0.3 (D)*, *D* denotes the default module. 
When loading packages, if you don't specify the version, the default module will be loaded. To load packages a user simply does:

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
^^^^^^^^^^^^^^^^^^^^^^^^

+------------------------------------+-------------------------------------------------+
| Command                            |          Effect                                 |
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