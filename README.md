# Progressivis

ProgressiVis is a Python toolkit and scientific workflow system that
implements a new _programming paradigm_ that we call _Progressive
Analytics_ aimed at performing analytics in a progressive way.  It
allows analysts to see the progress of their analysis and to steer it
while the computation is being done.

Instead of running algorithms to completion one after the other, as
done in all existing scientific analysis systems, ProgressiVis modules
run in short batches, each batch being only allowed to run for a
specific quantum of time - typically 1 second - producing a usable
result in the end, and yielding control to the next module.  To
perform the whole computation, ProxiVis loops over the modules as many
times as necessary to converge to a result that the analyst consider
satisfactory.

ProgressiVis relies on well known Python libraries, such as
[numpy](http://www.numpy.org/),[scipy](http://www.scipy.org/),
[Pandas](http://pandas.pydata.org/),
[Scikit-Learn](http://scikit-learn.org/), and
[Bokeh](http://bokeh.pydata.org/). 

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
