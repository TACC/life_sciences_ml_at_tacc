.. Intro to HPC @ TACC documentation master file, created by
   sphinx-quickstart on Fri Jun 26 14:44:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Your first steps at TACC
========================


Authentication 
^^^^^^^^^^^^^^
To participate in this workshop, you will need a TACC account and must have set up multi-factor authentication using a token app or SMS. 
You can do this by visiting the TACC portal https://tacc.utexas.edu/portal/login

.. image:: ./images/TACC_login.png
   :target: ./images/TACC_login.png
   :alt: TACC login page

Then clicking on your username at the top right of the page, 

.. image:: ./images/TACC_dashboard.png
   :target: ./images/TACC_dashboard.png
   :alt: TACC Dashboard
   :width: 800px


selecting "Manage Account", and, under MFA Pairing, clicking to pair. 

.. image:: ./images/TACC_MFA_pairing.png
   :target: ./images/TACC_MFA_pairing.png
   :alt: TACC MFA pairing
   :width: 800px


You can find more details about MFA Pairing `here <https://docs.tacc.utexas.edu/basics/mfa/>`_.

In your **TACC portal**, you can also view your allocations, open tickets, and the systems along with their current status.


HPC Systems
^^^^^^^^^^^
The training will be fully interactive. Participants are **strongly encouraged** to follow along on the command line.

In this workshop, we will use:

.. code-block:: console

   [local]$

for commands on the local system and:

.. code-block:: console

   [frontera]$ 

   [vista]$
   
for commands on the remote system consistently throughout.

Tips for Success
""""""""""""""""

Read the `documentation <https://docs.tacc.utexas.edu/>`_.

* Learn node schematics, limitations, file systems, rules
* Learn about the scheduler, queues, policies
* Determine the right resource for the job

User Responsibility on Shared Resources
"""""""""""""""""""""""""""""""""""""""

HPC systems are shared resources. Your jobs and activity on a cluster, if mismanaged,
can affect others. TACC staff are always `available to help <https://www.tacc.utexas.edu/about/help/>`_.
