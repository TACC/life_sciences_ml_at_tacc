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
^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

 <span style="text-align: justify;font-size: 16px;line-height:24px;">When you are finished with your job, you can end your job using the </span><b>End</b> 
 <span style="background-color:#FF7F00;color:white;"><b>&nbsp( 4 )&nbsp</b></span>   
 <span style="text-align: justify; font-size: 16px;line-height:24px;">button on the TAP Home Screen page or on the Job Status page. Note that closing the browser window will not end the job. Also note that if you end the job from within the application (for example, pressing "Enter" in the red xterm in a DCV or VNC job), TAP will still show the job as running until you check status for the job, click "End Job" within TAP, or the requested end time of the job is reached.</span>


Resubmitting a Past Job
^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^

.. raw:: html

    <span style="text-align: justify;font-size: 16px;line-height:24px;">TAP provides certain useful diagnostic and logistic utilities on the Utilities page. Access the Utilities page by selecting the <b>Utilities</b> <span    style="background-color:#FF7F00; color:#FFFFFF;"><b>&nbsp( 9 ) </b></span> &nbsp;button on the Home Screen page. 


.. image::  ./images/4TAP.png
   :target: ./images/3TAP.png
   :align: center
   :alt:  e 4. TAP Utilities



Configuring Jupyter Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Utilities section provides access to several common actions related to Jupyter Notebooks. 
* **"Use Python3"** sets the TACC Python3 module as part of your default modules so that TAP will use Python3 for Jupyter Notebooks. If you want to use a non-default Python installation, such as Conda, you will need to install it yourself via the system command line. TAP will use the first "jupyter-notebook" command in your `$PATH`, so make sure that the command "which jupyter-notebook" returns the Jupyter Notebook you want to use. Conda install typically configures your environment so that Conda is first on your `$PATH`.

"Link `$WORK` from `$HOME`" and "Link `$SCRATCH` from `$HOME`" create symbolic links in your `$HOME` directory so that you can access `$WORK` and `$SCRATCH` from within a Jupyter Notebook. TAP launches Jupyter Notebooks from within your `$HOME` directory, so these other file systems are not reachable without such a linking mechanism. The links will show up as "WORK" and "SCRATCH" in the Jupyter file browser. You only need to create these links once and they will remain available for all future jobs.

Obtaining TACC Account Status 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Status section provides system information and diagnostics. "Check TACC Info" will show account balances and filesystem usage for the selected system. "Run Sanity Tool" performs a series of sanity checks to catch common account issues that can impact TAP jobs (for example, being over filesystem quota on your `$HOME` directory).

Setting a Remote Desktop to Full Screen Mode  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^

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



Using Jupyter Notebooks
-----------------------

.. warning::

   Add section here on using Jupyter Notebooks for basic functions.

  



Jupyter Notebook user interface
-------------------------------

When you open a new Jupyter notebook, you will be presented with the
**notebook name**, a **menu bar**, a **toolbar** and an empty **code
cell**.

.. image::  ./images/Jupyter_start.png
   :target: ./images/Jupyter_start.png
   :alt:  Figure 9. Visualization of a Jupyter notebook.


**Notebook name**: The name displayed at the top of the page, next to
the Jupyter logo, reflects the name of the ``.ipynb`` file. Clicking on
the notebook name brings up a dialog which allows you to rename it.
 
**Menu bar**: The menu bar presents different options that may be used to manipulate the way the notebook functions.

**Toolbar**: The tool bar gives a quick way of performing the most-used operations within the notebook, by clicking on an icon.

**Code cell**: the default type of cell; read on for an explanation of cells.

Structure of a notebook document
--------------------------------

..
  :kbd:`Control-x Control-f`
  :kbd:`C-x C-f`
  :guilabel:`&Cancel`

The notebook consists of a sequence of cells. A cell is a multiline text
input field, and its contents can be executed by using ``Shift-Enter``, or by clicking either the "Play" button the
toolbar, or ``Cell`` and then ``Run`` in the menu bar.
The execution behavior of a cell is determined by the cell's type. There
are three types of cells: **code cells**, **markdown cells**, and **raw
cells**. Every cell starts off being a **code cell**, but its type can
be changed by using a drop-down on the toolbar (which will be "Code",
initially), or via {ref}\ ``keyboard shortcuts <keyboard-shortcuts>``.

For more information on the different things you can do in a notebook,
see the `collection of
examples <https://nbviewer.jupyter.org/github/jupyter/notebook/tree/main/docs/source/examples/Notebook/>`__.

Code cells
~~~~~~~~~~

A *code cell* allows you to edit and write new code, with full syntax
highlighting and tab completion. The programming language you use
depends on the *kernel*, and the default kernel (IPython) runs Python
code.

When a code cell is executed, code that it contains is sent to the
kernel associated with the notebook. The results that are returned from
this computation are then displayed in the notebook as the cell's
*output*. The output is not limited to text, with many other possible
forms of output are also possible, including ``matplotlib`` figures and
HTML tables (as used, for example, in the ``pandas`` data analysis
package). This is known as IPython's *rich display* capability.

.. code:: {seealso}

   [Rich Output] example notebook

Markdown cells
~~~~~~~~~~~~~~

You can document the computational process in a literate way,
alternating descriptive text with code, using *rich text*. In IPython
this is accomplished by marking up text with the Markdown language. The
corresponding cells are called *Markdown cells*. The Markdown language
provides a simple way to perform this text markup, that is, to specify
which parts of the text should be emphasized (italics), bold, form
lists, etc.

If you want to provide structure for your document, you can use markdown
headings. Markdown headings consist of 1 to 6 hash # signs ``#``
followed by a space and the title of your section. The markdown heading
will be converted to a clickable link for a section of the notebook. It
is also used as a hint when exporting to other document formats, like
PDF.

When a Markdown cell is executed, the Markdown code is converted into
the corresponding formatted rich text. Markdown allows arbitrary HTML
code for formatting.

Within Markdown cells, you can also include *mathematics* in a
straightforward way, using standard LaTeX notation: ``$...$`` for inline
mathematics and ``$$...$$`` for displayed mathematics. When the Markdown
cell is executed, the LaTeX portions are automatically rendered in the
HTML output as equations with high quality typography. This is made
possible by `MathJax <https://www.mathjax.org/>`__, which supports a
`large
subset <https://docs.mathjax.org/en/latest/input/tex/index.html>`__ of
LaTeX functionality

Standard mathematics environments defined by LaTeX and AMS-LaTeX (the
``amsmath`` package) also work, such as
``\begin{equation}...\end{equation}``, and
``\begin{align}...\end{align}``. New LaTeX macros may be defined using
standard methods, such as ``\newcommand``, by placing them anywhere
*between math delimiters* in a Markdown cell. These definitions are then
available throughout the rest of the IPython session.

.. code:: {seealso}

   [Working with Markdown Cells] example notebook

Raw cells
~~~~~~~~~

*Raw* cells provide a place in which you can write *output* directly.
Raw cells are not evaluated by the notebook. When passed through
[nbconvert], raw cells arrive in the destination format unmodified. For
example, you can type full LaTeX into a raw cell, which will only be
rendered by LaTeX after conversion by nbconvert.

Basic workflow
--------------

The normal workflow in a notebook is, then, quite similar to a standard
IPython session, with the difference that you can edit cells in-place
multiple times until you obtain the desired results, rather than having
to rerun separate scripts with the ``%run`` magic command.

Typically, you will work on a computational problem in pieces,
organizing related ideas into cells and moving forward once previous
parts work correctly. This is much more convenient for interactive
exploration than breaking up a computation into scripts that must be
executed together, as was previously necessary, especially if parts of
them take a long time to run.

To interrupt a calculation which is taking too long, use the
{guilabel}\ ``Kernel``, {guilabel}\ ``Interrupt`` menu option, or the
{kbd}\ ``i,i`` keyboard shortcut. Similarly, to restart the whole
computational process, use the {guilabel}\ ``Kernel``,
{guilabel}\ ``Restart`` menu option or {kbd}\ ``0,0`` shortcut.

A notebook may be downloaded as a ``.ipynb`` file or converted to a
number of other formats using the menu option {guilabel}\ ``File``,
{guilabel}\ ``Download as``.

.. code:: {seealso}

   [Running Code in the Jupyter Notebook] example notebook

   [Notebook Basics] example notebook

(keyboard-shortcuts)=

Keyboard shortcuts
~~~~~~~~~~~~~~~~~~

All actions in the notebook can be performed with the mouse, but
keyboard shortcuts are also available for the most common ones. The
essential shortcuts to remember are the following:

-  {kbd}\ ``Shift-Enter``: run cell : Execute the current cell, show any
   output, and jump to the next cell below. If {kbd}\ ``Shift-Enter`` is
   invoked on the last cell, it makes a new cell below. This is
   equivalent to clicking the {guilabel}\ ``Cell``, {guilabel}\ ``Run``
   menu item, or the Play button in the toolbar.
-  {kbd}\ ``Esc``: Command mode : In command mode, you can navigate
   around the notebook using keyboard shortcuts.
-  {kbd}\ ``Enter``: Edit mode : In edit mode, you can edit text in
   cells.

For the full list of available shortcuts, click {guilabel}\ ``Help``,
{guilabel}\ ``Keyboard Shortcuts`` in the notebook menus.

Searching
---------

Jupyter Notebook has an advanced built-in search plugin for finding text
within a notebook or other document, which uses the {kbd}\ ``Ctrl-F``
({kbd}\ ``Cmd+F`` for macOS) shortcut by default.

Your browser's ``find`` function will give unexpected results because it
doesn't have access to the full content of a document (by default), but
you can still use your browser find function from the browser menu if
you want, or you can disable the built-in search shortcut using the
Advanced Settings Editor.

Alternatively, you can disable windowed notebook rendering to expose the
full document content to the browser at the expense of performance.

Plotting
--------

One major feature of the Jupyter notebook is the ability to display
plots that are the output of running code cells. The IPython kernel is
designed to work seamlessly with the [matplotlib] plotting library to
provide this functionality. Specific plotting library integration is a
feature of the kernel.

Installing kernels
------------------

For information on how to install a Python kernel, refer to the `IPython
install page <https://ipython.org/install.html>`__.

The Jupyter wiki has a long list of `Kernels for other
languages <https://github.com/jupyter/jupyter/wiki/Jupyter-kernels>`__.
They usually come with instructions on how to make the kernel available
in the notebook.

(signing-notebooks)=

Trusting Notebooks
------------------

To prevent untrusted code from executing on users' behalf when notebooks
open, we store a signature of each trusted notebook. The notebook server
verifies this signature when a notebook is opened. If no matching
signature is found, Javascript and HTML output will not be displayed
until they are regenerated by re-executing the cells.

Any notebook that you have fully executed yourself will be considered
trusted, and its HTML and Javascript output will be displayed on load.

If you need to see HTML or Javascript output without re-executing, and
you are sure the notebook is not malicious, you can tell Jupyter to
trust it at the command-line with:

::

   $ jupyter trust mynotebook.ipynb

See the `security
documentation <https://jupyter-server.readthedocs.io/en/stable/operators/security.html>`__
for more details about the trust mechanism.

Browser Compatibility
---------------------

The Jupyter Notebook aims to support the latest versions of these
browsers:

-  Chrome
-  Safari
-  Firefox

Up to date versions of Opera and Edge may also work, but if they don't,
please use one of the supported browsers.

Using Safari with HTTPS and an untrusted certificate is known to not
work (websockets will fail).

.. code:: {eval-rst}

   .. include:: links.txt



