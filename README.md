# Progressivis

[![Build Status](https://travis-ci.org/jdfekete/progressivis.svg?branch=master&label=Travis%20CI)](https://travis-ci.org/jdfekete/progressivis)

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


## Installation

ProgressiVis can be installed with pip with or without virtualenv.
From a virtualenv or from the global environment, install it with:
```
pip install -e requirements.txt
python setup.py install
```

or, with anaconda:

```
conda config --add channels progressivis
conda install progressivis
```

## Examples

To see examples, either look at the tests in the `tests` directory, or
try the examples in the `examples` directory.

## Live demo

A live is available online: as **Jupyter notebook** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jdfekete/progressivis.git/master?filepath=notebooks%2FPsBoardDemo4Binder.ipynb)
<!--* as **Jupyter voilà dashboard** [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jdfekete/progressivis.git/master?filepath=%2F..%2Fvoila%2Frender%2Fnotebooks%2FPsBoardDemo4Binder.ipynb)-->



## Contribute

## Support

If you are having issues, please let us know at [issue](https://github.com/jdfekete/progressivis/issues).


## License

The project is licensed under the BSD license.

