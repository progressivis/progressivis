# Installation of ProgressiVis

## Installation for users

If you only need the `ProgressiVis` core, install the `progressivis` package, without options. If you need `ProgressiVis` with it's `Jupyter` visualizations or if you don't know all your needs in advance install  `progressivis[jupyter]`.

Currently these installations are continually tested with `Linux`. They should also work on `MacOS` and `Windows`.

### With pip

#### Installing `ProgressiVis` core:

```
pip install progressivis
```

(progressivis-jupyter-install)=

#### Installing `ProgressiVis` with it's `Jupyter` extension:

```
pip install progressivis[jupyter]
```

## Installation for developers

```{eval-rst}
.. note::
   To fully utilize all the capabilities of `ProgressiVis`, including visualizations, you must install both the `progressivis` repository and it's `Jupyter` extension repository (named `ipyprogressivis`), as outlined below. It’s recommended to clone both repositories into a shared directory called `$WORKDIR` below.
```

### Installing `ProgressiVis`

1. Clone the [ProgressiVis repository](https://github.com/progressivis/progressivis/):

```
cd $WORKDIR
git clone https://github.com/progressivis/progressivis.git
```

It is not mandatory, but it is preferable to install `ProgressiVis` in a dedicated `Python` environment, created for example with `conda`. If you don't want to work in a `conda` environment, skip the following 2 steps.

2. Create a `conda` environment (named below `pvenv`):

```
conda create -n pvenv python=3.13
```


3. Activate this environment:

```
conda activate pvenv
```
4. Execute:

```
cd $WORKDIR/progressivis
pip install -e .[typing]
```
#### Getting updates

```
cd $WORKDIR/progressivis
git pull
```

### Installing the `Jupyter` extension for `ProgressiVis`:

1. Clone the repository named [ipyprogressivis](https://github.com/progressivis/ipyprogressivis/):

```
cd $WORKDIR
git clone https://github.com/progressivis/ipyprogressivis.git
```

2. If you choose to work in a conda environment, activate the environment you have created for  `progressivis` (previously called `pvenv`).

3. Install `yarn v1` in your environment

If you are using a conda environment do:

```
conda install yarn=1
```

Otherwise use the means specific to your environment to install `yarn v1`

4. Execute:

```
cd $WORKDIR/ipyprogressivis
pip install -e .[dev]
```

#### Getting updates

```
cd $WORKDIR/ipyprogressivis
git pull
```

then:

```
cd $WORKDIR/ipyprogressivis/ipyprogressivis/js
yarn run build
```

If `jupyter` is running, you should restart it.

```{eval-rst}
.. note::
   At this stage, the user guide examples should work in the jupyter lab environment. `ProgressiVis` notebook relies on jupyter lab rather than the traditional notebook because progressive programs are not linear by nature and need special navigation tools in the notebook, easier to design and deploy on jupyter lab.
```