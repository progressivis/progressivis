# Introduction

ProgressiVis is a system and language implementing *progressive data
analysis and visualization*.

In the ProgressiVis language, all the executions are progressive by
design. It also implements extensions in the notebook to create
interactive visualizations and their user interfaces for controlling
the progressive exploration.

When visualizing the results of computations, the visualizations are
shown, updated, and improved progressively, every few seconds, until
the final result is computed. Alternatively, the user can abort the
computation if it does not converge to the desired result.

## Why?

Interactive data exploration is performed by humans and therefore
requires a controlled latency. When it exceeds 10s, humans cannot
maintain their attention and their effectiveness drops dramatically.
Instead of running algorithms to completion one after the other, as
done in all existing scientific analysis systems, ProgressiVis
algorithms (called modules) run in short batches, each batch being
only allowed to run for a specific quantum of time - typically 1
second - producing a usable result in the end, and yielding control to
the next module.  To perform the whole computation, ProgressiVis loops
over the modules as many times as necessary to converge to a result
that the analyst considers satisfactory.

Humans can then conduct data exploration using large datasets and
powerful analysis algorithms, trading time with quality, remaining in
control of the quality they need to make decisions by controlling the
time they will leave the algorithm run.


ProgressiVis relies on well known Python libraries, such as
[numpy](http://www.numpy.org/),[scipy](http://www.scipy.org/),
[Pandas](http://pandas.pydata.org/),
[pyarrow](https://arrow.apache.org/docs/python/index.html),
and
[Scikit-Learn](http://scikit-learn.org/).

For now, ProgressiVis is mostly a proof of concept. The current
implementation provides progressive data structures that can grow and
adapt to progressive computation, an execution model relying on
asynchronous programming with non-preemptive [cooperative
multitasking](https://en.wikipedia.org/wiki/Cooperative_multitasking),
and a number of *modules* implementing various components of the
language: data manipulation, computations, statistics, machine
learning, and vizualization.  ProgressiVis is also meant to be used
from jupyter notebooks; it provides interactions and visualizations
specially suited to progressive systems.
