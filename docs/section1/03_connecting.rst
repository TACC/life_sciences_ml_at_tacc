Your First Steps at TACC
========================

Connecting to the Machines
^^^^^^^^^^^^^^^^^^^^^^^^^^
 
Connecting to Frontera
----------------------

To log in to Frontera, follow the instructions for your operating system below.

Mac / Linux
^^^^^^^^^^^

Open the application 'Terminal'

.. code-block:: console
   
   [local]$ ssh username@frontera.tacc.utexas.edu
   (enter password)
   (enter 6-digit token)

Windows
^^^^^^^

Windows users will need to install an SSH client like **PuTTY** to follow along. If you
have not done so already, download the **PuTTY** "Windows Installer"
`here <https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html>`_. (Other tools like
PowerShell work, too).

Once **PuTTY** is installed:

* Double-click the **PuTTY** icon
* In the **PuTTY** Configuration window make sure the Connection type is SSH
* enter ``frontera.tacc.utexas.edu`` for Host Name
* click "Open"
* answer "Yes" to the SSH security question

In the **PuTTY** terminal:

* enter your TACC user id after the "login as:" prompt, then Enter
* enter the password associated with your TACC account
* enter your 6-digit TACC token value

.. code-block:: console

   Open the application 'PuTTY'
   enter Host Name: frontera.tacc.utexas.edu
   (click 'Open')
   (enter username)
   (enter password)
   (enter 6-digit token)

Successful Login to Frontera
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your login was successful, your terminal will look something like this:


.. code-block:: console 

   ------------------------------------------------------------------------------
                      Welcome to the Frontera Supercomputer
         Texas Advanced Computing Center, The University of Texas at Austin
   ------------------------------------------------------------------------------

                 ** Unauthorized use/access is prohibited. **
   
   If you log on to this computer system, you acknowledge your awareness
   of and concurrence with the UT Austin Acceptable Use Policy. The
   University will prosecute violators to the full extent of the law.
   
   TACC Usage Policies:
   http://www.tacc.utexas.edu/user-services/usage-policies/
   ______________________________________________________________________
   
   Welcome to Frontera, please read these important system notes:
   
   --> Frontera user documentation is available at:
          https://portal.tacc.utexas.edu/user-guides/frontera
   
   ----------------------- Project balances for user lima ------------------------
   | Name           Avail SUs     Expires |                                      |
   | TACC-SCI          102149  2025-06-30 |                                      |
   -------------------------- Disk quotas for user lima --------------------------
   | Disk         Usage (GB)     Limit    %Used   File Usage       Limit   %Used |
   | /home1              0.7      25.0     2.90         4752      200000    2.38 |
   | /work2             68.4    1024.0     6.68       232978     3000000    7.77 |
   | /scratch1           0.0       0.0     0.00            7           0    0.00 |
   | /scratch2           0.0       0.0     0.00            1           0    0.00 |
   | /scratch3           0.0       0.0     0.00            1           0    0.00 |
   -------------------------------------------------------------------------------
   
   Tip 131   (See "module help tacc_tips" for features or how to disable)

      Use Ctrl+E to go the end of the command line.

A Note About Quotas
^^^^^^^^^^^^^^^^^^^

The welcome message you receive upon successful login to Frontera has useful information
for you to keep track of. Especially of note is the breakdown of disk quotas for your account,
as you can keep an eye on whether your usage is nearing the determined limit. 

Once your usage is nearing the quota, you'll start to experience issues that will not only
impact your own work, but also impact the system for others. For example, if you're nearing
your quota in ``$WORK``, and your job is repeatedly trying (and failing) to write to ``$WORK``,
you will stress that file system.

``$HOME``

* is backed up
* Recommended Use: cron jobs, scripts and templates, environment settings, compilations

``$WORK``

* NOT backed up
* Recommended Use: software installations, original datasets that can't be reproduced.

``$SCRATCH``

* NOT backed up, subject to purge if access time is more than 10 days old
* Recommended Use: Reproducible datasets, I/O files: temporary files, checkpoint/restart files, job output files

Another useful way to monitor your disk quotas (and TACC project balances) at any time is to execute:

.. code-block:: console

   [frontera]$ /usr/local/etc/taccinfo


