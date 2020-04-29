# progressivis_nb_widgets: A Custom Jupyter Widget Library for Progressivis

## Installation (by now, dev mode only)


### Development installation for notebook (requires npm),

    $ git clone "https://github.com/jdfekete/progressivis.git
    $ pip install -e .
    $ cd progressivis_nb_widgets/js
    $ npm install 
    $ npm run build
    $ jupyter nbextension install --py --symlink --sys-prefix progressivis.progressivis_nb_widgets.nbwidgets
    $ jupyter nbextension enable --py --sys-prefix progressivis.progressivis_nb_widgets.nbwidgets

### Development installation for jupyterlab

In addition to previous steps (for notebook) 

    $ cd progressivis_nb_widgets
    $ jupyter labextension install js


NB: When actively developing your extension, build Jupyter Lab with the command:

    $ jupyter lab --watch


## Running under Voilà

### As a standalone app :

voila --enable_nbextensions=True

### As a server extension to notebook or jupyter_server

First, you have to enable the extension:

$ jupyter serverextension enable voila --sys-prefix


Then run:

$jupyter lab --VoilaConfiguration.enable_nbextensions=True

or

$jupyter notebook --VoilaConfiguration.enable_nbextensions=True

When running the Jupyter server, the Voilà app is accessible from the base url suffixed with voila

### Using a JupyterLab extension to render a notebook with voila

Install the extension :

jupyter labextension install @jupyter-voila/jupyterlab-preview

Display the notebook with Voilà like showed here: https://user-images.githubusercontent.com/591645/59288034-1f8d6a80-8c73-11e9-860f-c3449dd3dcb5.gif
