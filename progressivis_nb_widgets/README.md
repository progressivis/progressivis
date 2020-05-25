# progressivis_nb_widgets: A Custom Jupyter Widget Library for Progressivis

## Installation (by now, dev mode only)
NB: You are assumed to have previously installed *progressivis* library
The root of the following relative path is the progressivis repository root

### Preparing conda environment
* Start by installing miniconda, following Conda’s installation documentation :
https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
* Create a conda environment [first time] :
  $ conda create -n progressivis-ext --override-channels --strict-channel-priority -c conda-forge -c anaconda cython numpy jupyterlab nodejs git
* Activate it :
  $ conda activate progressivis-ext
* Install jupyterlab=2 and ipywidgets
  $ conda install -c conda-forge jupyterlab=2
  $ conda install -c conda-forge ipywidgets

### Development installation for notebook (requires npm),
    $ cd progressivis_nb_widgets
    $ pip install -e . 
    $ cd progressivis_nb_widgets/js
    $ npm install 
    $ npm run build
    $ jupyter nbextension install --py --symlink --sys-prefix progressivis_nb_widgets.nbwidgets
    $ jupyter nbextension enable --py --sys-prefix progressivis_nb_widgets.nbwidgets

### Development installation for jupyterlab

NB: be sure to run jupyterlab2. If needed, create a dedicated environment as explained here : 
https://jupyterlab.readthedocs.io/en/stable/developer/extension_tutorial.html#set-up-a-development-environment

In addition to previous steps (i.e. notebook steps) 

    $ cd progressivis_nb_widgets
    $ jupyter labextension install js





## Running examples

NB: Before running examples, activate progressivis-ext :

## Running under notebook

$ cd progressivis/notebooks
$ jupyter notebook

## Running under jupyterlab

$ cd progressivis/notebooks
$ jupyter lab

NB: install and enable @jupyter-widgets/jupyterlab-manager
NB: in order to run progressivis examples you have to enable third-party extensions in jupyterlab extension manager
NB: When actively developing your extension, build Jupyter Lab with the command:
$ jupyter lab --watch
$ conda activate progressivis-ext

### Running under Voilà

Install voilà :

$ conda install -c conda-forge voila

#### As a standalone app :

voila --enable_nbextensions=True YourNotebook.ipynb

#### As a server extension to notebook or jupyter_server

First, you have to enable the extension:

$ jupyter serverextension enable voila --sys-prefix


Then run:

$jupyter lab --VoilaConfiguration.enable_nbextensions=True

or

$jupyter notebook --VoilaConfiguration.enable_nbextensions=True

When running the Jupyter server, the Voilà app is accessible from the base url suffixed with voila

#### Using a JupyterLab extension to render a notebook with voila

Install the extension :

jupyter labextension install @jupyter-voila/jupyterlab-preview

Display the notebook with Voilà like showed here: https://user-images.githubusercontent.com/591645/59288034-1f8d6a80-8c73-11e9-860f-c3449dd3dcb5.gif
