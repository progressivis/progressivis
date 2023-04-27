```{eval-rst}
.. progressivis documentation master file, created by
   sphinx-quickstart on Fri Feb 16 00:36:48 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
```

# Welcome to ProgressiVis

![Image of the progressive construction of the Eiffel Tower](construction_tour_eiffel.jpg "Progressive construction of the Eiffel Tower")

ProgressiVis is a system or language implementing *progressive data analysis and visualization*. Data exploration requires a controlled latency; when it exceeds 10s, humans cannot maintain their attention and their effectiveness drops dramatically.  Instead of performing long/unbounded computations, ProgressiVis quickly returns an approximate result that improves over time, *progressively*.

When visualizing the results of computations, the visualizations are shown, updated, and improved progressively, every few seconds, until the final result is computed. Alternatively, the user can abort the computation if it does not converge to the desired result.

ProgressiVis implements a progressive language where all the executions are progressive by design. It also implements extensions in the notebook to create interactive visualizations and their user interface for controlling the progressive exploration.


```{eval-rst}
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro
   install
   userguide
   reference
   module_library
   custom_modules
   chaining_widgets
```

# Indices and tables


```{eval-rst}
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```
