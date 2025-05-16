=======================
Life Sciences ML @ TACC
=======================

Quickstart
-----------

1. Create a Python 3 environment (virtual or with conda)
2. Install dependencies ``pip install -r requirements.txt``
3. Navigate to the ``docs`` folder
4. Build using ``make html`` (Mac/Linux) or ``make.bat html`` (Windows)
5. Use ``make livehtml`` to start a server that watches for source
   changes and will rebuild/refresh the docs automatically. Go to
   http://localhost:8000/ to see its output.


reStructuredText help
---------------------

rST is a bit more onerous than Markdown, but it includes more advanced features
like inter-page references/links and a suite of directives.

- `Sphinx's primer <http://www.sphinx-doc.org/en/stable/rest.html>`_
- `Full Docutils reference <http://docutils.sourceforge.net/rst.html>`_

  - also see its `Quick rST <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_ cheat sheet.

- `Online rST editor <https://feat.dlup.link/rsted>`_
- `Projects using Sphinx <https://www.sphinx-doc.org/en/master/examples.html>`_
