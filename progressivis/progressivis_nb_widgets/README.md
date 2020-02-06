progressivis_nb_widgets
===============================

A Custom Jupyter Widget Library for Progressivis

Installation
------------

To install use pip:

    $ pip install nbwidgets
    $ jupyter nbextension enable --py --sys-prefix nbwidgets

To install for jupyterlab

    $ jupyter labextension install nbwidgets

For a development installation (requires npm),

    $ git clone "https://github.com/jdfekete/progressivis.git
    $ pip install -e .
    $ cd progressivis_nb_widgets/js
    $ npm install 
    $ npm run build
    $ jupyter nbextension install --py --symlink --sys-prefix progressivis.progressivis_nb_widgets.nbwidgets
    $ jupyter nbextension enable --py --sys-prefix progressivis.progressivis_nb_widgets.nbwidgets
    $ # for jupyterlab (not working yet)
    $ cd progressivis_nb_widgets
    $ jupyter labextension install js

When actively developing your extension, build Jupyter Lab with the command:

    $ jupyter lab --watch # not working yet

This take a minute or so to get started, but then allows you to hot-reload your javascript extension.
To see a change, save your javascript, watch the terminal for an update.

Note on first `jupyter lab --watch`, you may need to touch a file to get Jupyter Lab to open.

