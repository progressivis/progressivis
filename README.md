# ProgressiVis

[![Python Tests](https://github.com/progressivis/progressivis/actions/workflows/python.yml/badge.svg?branch=master&event=push)](https://github.com/progressivis/progressivis/actions/workflows/python.yml)
[![Documentation Status](https://readthedocs.org/projects/progressivis/badge/?version=latest)](https://progressivis.readthedocs.io/en/latest/?badge=latest)
[![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy)
[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

ProgressiVis is a Python toolkit and scientific workflow system that implements a new [programming paradigm](https://en.wikipedia.org/wiki/Programming_paradigm) that we call _Progressive Data Analysis_  (PDA), aimed at performing exploratory data analysis and visualization in a progressive way.  It allows analysts to visualize the progress of their analysis and to steer it while the computation is being done. See our book on [Progressive Data Analysis](https://www.aviz.fr/Progressive/PDABook).

Instead of running a pipeline of algorithms to completion, one after the other, as done in all existing scientific analysis systems, ProgressiVis modules run in short batches, each batch being only allowed to run for a specific quantum of time&mdash; typically 0.5 second&mdash; producing a usable result in the end, and yielding control to the next module.  To perform the whole computation, ProgressiVis loops over the modules as many times as necessary to converge to a result that the analyst considers satisfactory.

In addition to running data science pipelines progressively, ProgressiVis opens the door to interactive machine learning (ML), green computing, and a certain form of algorithm explainability.
It is a step towards green computing by stopping expensive analyses on large datasets as soon as a useful result is computed, instead of processing them completely.
It provides algorithm transparency, a certain form of explainability, since progressive visualization reveals how algorithms transform data as they run.

Unfortunately, the cost of building progressive applications is very high due to the lack of libraries or languages supporting its paradigm. 
The ProgressiVis toolkit is designed to allow practitioners and researchers to enter the PDA world with reasonable effort.

ProgressiVis relies on well-known Python libraries, such as
[numpy](http://www.numpy.org/),[scipy](http://www.scipy.org/),
[Pandas](http://pandas.pydata.org/),
and
[Scikit-Learn](http://scikit-learn.org/).

For now, ProgressiVis is mostly a proof of concept. You can find its documentation [here](https://progressivis.readthedocs.io/en/latest/).

## Installation

See the installation instructions [provided here](https://progressivis.readthedocs.io/en/latest/install.html).

## Examples

To see examples, either look at the tests in the `tests` directory or try the examples in the `notebooks` directory.

## Running demos (on your computer)

ProgressiVis demos require visualisations which are available in the `progressivis` extension called `ipyprogressivis`. Please follow the instructions provided [here](https://github.com/progressivis/ipyprogressivis)


## Support

If you are having issues, please let us know at [issue](https://github.com/progressivis/progressivis/issues).


## License

The project is licensed under the BSD-2-Clause license.

