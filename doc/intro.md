# Introduction

ProgressiVis is a Python toolkit implementing
*Progressive Data Analysis*. A book describes the paradigm in details, see
[www.aviz.fr/Progressive/PDABook](https://www.aviz.fr/Progressive/PDABook).

In the `ProgressiVis` toolkit, all the executions are progressive by
design; the system is never blocked performing lengthy operations, it
always shows visualizations quickly, even when all the data is not
loaded yet or the algorithm execution is not finished.
`ProgressiVis` also comes with extensions in the notebook
to create interactive visualizations and their user interfaces for
controlling the progressive process.

When visualizing the results of computations, the visualizations are
shown, updated, and improved progressively, every few seconds, until
the final result is computed. Alternatively, the user can abort the
current computation and try a new one or several, if the current one
does not converge to the expected result.

## Why?

Interactive data exploration is performed by humans and therefore
requires a controlled latency. When the latency exceeds 10s, humans
cannot maintain their attention and their efficiency at exploring data
drops dramatically.  Instead of loading data fully or running
algorithms to completion one after the other, as done in all existing
scientific analysis systems, `ProgressiVis` algorithms (called modules)
run in short batches, each batch being only allowed to run for a
specific quantum of time - typically 1 second - producing a usable
result in the end, and yielding control to the next module.  To
perform the whole computation, `ProgressiVis` loops over the modules as
many times as necessary to converge to a result that the analyst
considers satisfactory.

Humans can then conduct interactive data exploration using large
datasets and powerful analysis algorithms, trading time with quality,
staying in control of the quality they need to make decisions by
controlling the time they will allow the algorithm to run.

## Benefits

Progressive Data Analysis provides the following benefits:

- **Scalability for visualization** in terms of data size and download time
- **Scalability for interactive analysis** including machine learning
- **Instant data**, no need to wait for data and visualization to arrive
- **Green computing**, processing only the required data to get a result
- **Algorithmic transparency**, the possibility to monitor and visualize data processing as it runs.

## Current State

For now, `ProgressiVis` is mostly a prototype. The current
implementation provides progressive **data structures** that can grow and
adapt to progressive computations, an **execution model** relying on
asynchronous programming with non-preemptive [cooperative
multitasking](https://en.wikipedia.org/wiki/Cooperative_multitasking),
and a number of **modules** implementing various components of the
language: data manipulation, computations, statistics, machine
learning, and vizualization.  ProgressiVis is also meant to be used
from jupyter notebooks; it provides interactions and visualizations
specially suited to progressive systems.

The authors are committed to maintain and improve `ProgressiVis` to become a mature toolkit.


## Dependencies

`ProgressiVis` relies on well known Python libraries, such as
[numpy](http://www.numpy.org/),[scipy](http://www.scipy.org/),
[Pandas](http://pandas.pydata.org/),
[pyarrow](https://arrow.apache.org/docs/python/index.html),
and
[Scikit-Learn](http://scikit-learn.org/).

