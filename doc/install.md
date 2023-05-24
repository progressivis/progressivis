# Installation of Progressivis

## Installation for users

### With miniconda/anaconda (recommended):

Currently, the easiest way to install *ProgressiVis* is as follows:

1. Install the latest version of
   [miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Install the progressivis package

In the future, you should be able to install ProgressiVis with:

```
conda install progressivis
```

### With pip

In the future, you should be able to install ProgressiVis with:
```
pip install progressivis
```


## Installation for developers

1. Clone the progressivis repository from [github.com](https://github.com/progressivis/progressivis/) along with its submodules, with the command:

```
git clone --recurse-submodules https://github.com/progressivis/progressivis.git
```

2. Create a conda environment with the following command:

```
conda env create -f environment.yml
```
It will create an environment called *progressivis*.

3. Activate this environment:

```
conda activate progressivis
```
4. Execute the following commands:
```
pip install -e .
pip install -e widgets
```

5. Then, install the jupyter notebook extensions with the following commands:

```
jupyter nbextension install --py --symlink --sys-prefix progressivis_nb_widgets.nbwidgets
jupyter nbextension enable --py --sys-prefix progressivis_nb_widgets.nbwidgets
```

Or, if you use jupyterlab:

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @jupyter-widgets/jupyterlab-sidecar
jupyter labextension install jupyterlab-datawidgets
jupyter labextension install widgets/progressivis_nb_widgets/js
```

## With pip

ProgressiVis can be installed with pip with or without virtualenv.
It requires a few system components. For example, on Ubuntu linux, you need to insall the packages `git` and `build-essential`.
Then, install ProgressiVis with:
```
pip install -r requirements.txt
python setup.py install
```
