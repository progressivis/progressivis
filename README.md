# Progressivis

[![Python Tests](https://github.com/progressivis/progressivis/actions/workflows/python.yml/badge.svg?branch=master&event=push)](https://github.com/progressivis/progressivis/actions/workflows/python.yml)
[![Documentation Status](https://readthedocs.org/projects/progressivis/badge/?version=latest)](https://progressivis.readthedocs.io/en/latest/?badge=latest)

ProgressiVis is a Python toolkit and scientific workflow system that
implements a new [programming
paradigm](https://en.wikipedia.org/wiki/Programming_paradigm) that we
call _Progressive Analytics_ aimed at performing analytics in a
progressive way.  It allows analysts to see the progress of their
analysis and to steer it while the computation is being done. See the
[workshop paper](https://hal.inria.fr/hal-01202901).

Instead of running algorithms to completion one after the other, as
done in all existing scientific analysis systems, ProgressiVis modules
run in short batches, each batch being only allowed to run for a
specific quantum of time - typically 1 second - producing a usable
result in the end, and yielding control to the next module.  To
perform the whole computation, ProgressiVis loops over the modules as many
times as necessary to converge to a result that the analyst considers
satisfactory.

ProgressiVis relies on well known Python libraries, such as
[numpy](http://www.numpy.org/),[scipy](http://www.scipy.org/),
[Pandas](http://pandas.pydata.org/),
and
[Scikit-Learn](http://scikit-learn.org/).

For now, ProgressiVis is mostly a proof of concept. It has bugs, but
more importantly, the standard Python libraries are not well-suited to
progressive execution. In particular, Numpy/SciPy/Pandas are not good
at growing arrays/DataFrames dynamically, they require the whole array
to be reconstructructed from scratch. This reconstruction is extremely
costly currently, but could become almost acceptable with some
internal changes.  The current implementation provides replacements data
structures that can grow and adapt to progressive computation.

## Live demos

Many interactive demos are available online. They will run remotely on [mybinder.org](https://mybinder.org/) so you can experiment with Progressivis with no need to install it locally.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/progressivis/progressivis.git/stable?filepath=demos)

Click the "launch binder" image above to run the the live demos on [mybinder.org](https://mybinder.org/).

**NB:** Interactive demos may take up to several minutes to build depending on the server load.

## Installation

### With miniconda/anaconda (recommended):

Currently, the easiest way to install *progressivis* is as follows:

1. Install the latest version of miniconda (if not yet done)

2. Clone the progressivis repository from [github.com](https://github.com/progressivis/progressivis/) along with its submodules with the command:

```
git clone --recurse-submodules https://github.com/progressivis/progressivis.git
```

3. Create a conda environment with the following command:

NB: by default it will create an environment called *progressivis*. If you want, you can change this name in the file *environment.yml* before runninng the command. Remember to reflect this change in the following commands.

```
conda env create -f environment.yml
```
3. Activate this environment:

```
conda activate progressivis
```
4. Execute the following commands:

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

### With pip (without the Jupyter interface)

ProgressiVis can be installed with pip with or without virtualenv.
From a virtualenv or from the global environment, install it with:
```
pip install -e requirements.txt
python setup.py install
```


## Examples

To see examples, either look at the tests in the `tests` directory, or
try the examples in the `examples` directory.

## Running demos (on your computer)

After installing progressivis **with miniconda/anaconda**, do:
```
cd notebooks
jupyter notebook
```
then run the notebooks of your choice

## Contribute

## Support

If you are having issues, please let us know at [issue](https://github.com/progressivis/progressivis/issues).


## License

The project is licensed under the BSD license.

