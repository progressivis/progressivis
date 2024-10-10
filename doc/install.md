# Installation of Progressivis

## Installation for users

If you only need the *Progressivis* core, install the `progressivis` package. If you need *Progressivis* with visualizations or if you don't know all your needs in advance install  `ipyprogressivis` (which will also install `progressivis` as a dependency). Installation procedures are similar between `progressivis` and `ipyprogressivis`, so in the following `[ipy]progressivis` means `progressivis` or `ipyprogressivis`

Currently these installations have been tested only with *Linux*.



### With miniforge (recommended):

Currently, the easiest way to install *ProgressiVis* is as follows:

1. Install (if not installed yet) the latest version of [miniforge](https://github.com/conda-forge/miniforge). Optionally you can create and activate a dedicated [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or activate an existing one.

2. Install the progressivis/ipyprogressivis package:

```
mamba install [ipy]progressivis -c progressivis
```


### With miniconda/anaconda:

1. Install  (if not installed yet) the latest version of [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://www.anaconda.com/download). As explained previously for `miniforge` you can use a conda environment of your choice.
​
2. Install the progressivis/ipyprogressivis package:

```
conda install mamba -c conda-forge  # if not installed yet
mamba install [ipy]progressivis -c progressivis -c conda-forge
```

NB: If you prefer, you can omit the mamba installation and use conda instead.


### With pip

In the future, you should be able to install [Ipy]ProgressiVis with:
```
pip install [ipy]progressivis
```


## Installation for developers

NB: To fully utilize all the capabilities of ProgressiVis, including visualizations, you must install both rogressivis and ipyprogressivis repositories in that order, as outlined below. It’s recommended to clone both repositories into a shared directory.

### Installing ProgressiVis

1. Clone the progressivis repository from [github.com](https://github.com/progressivis/progressivis/) along with its submodules, with the command:

```
git clone --recurse-submodules https://github.com/progressivis/progressivis.git
# then
cd progressivis
```

It is not mandatory, but it is preferable to install progressivis in a dedicated python environment, created for example with `conda`. If you don't want to work in a conda environment, skip the following 2 steps.

2. Create a conda environment with the following command:

```
<<<<<<< Updated upstream
conda create -n myenv python=3.12 -c conda-forge
=======
conda create -n pv312 python=3.11 -c conda-forge
>>>>>>> Stashed changes
```


3. Activate this environment:

```
conda activate pv312
```
4. Execute the following command:

```
pip install -e .
```

### Installing IpyProgressiVis

1. Clone the progressivis repository from [github.com](https://github.com/progressivis/ipyprogressivis/) with the command:

```
git clone https://github.com/progressivis/ipyprogressivis.git
# then
cd ipyprogressivis
```

2. If you choose to work in a conda environment, follow the steps 2 and 3 described previously for `progressivis` to create and activate an environment.

3. Depending on your development goals, install `progressivis` for users or for developpement

4. Install `yarn v1` in your environment

```
conda install yarn=1 -c conda-forge
```



5. Execute the following command:

```
pip install -e .
```

