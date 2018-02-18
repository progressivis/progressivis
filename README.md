# Progressivis

[![Build Status](https://travis-ci.org/jdfekete/progressivis.svg?branch=master)](https://travis-ci.org/jdfekete/progressivis)

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
[Scikit-Learn](http://scikit-learn.org/), and
[Bokeh](http://bokeh.pydata.org/). 

For now, ProgressiVis is mostly a proof of concept. It has bugs, but
more importantly, the standard Python libraries are not well-suited to
progressive execution. In particular, Numpy/SciPy/Pandas are not good
at growing arrays/DataFrames dynamically, they require the whole array
to be reconstructructed from scratch. This reconstruction is extremely
costly currently, but could become almost acceptable with some
internal changes.


## Installation

ProgressiVis can be installed with pip with or without virtualenv.
From a virtualenv or from the global environment, install it with:
```
python setup.py install
```

## Examples

To see examples, either look at the tests in the `tests` directory, or
try the notebook examples in the `examples` directory.

## Contribute

## Support

If you are having issues, please let us know.


## License

The project is licensed under the BSD license.

