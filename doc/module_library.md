# Module Library

## I/O Modules

### Basic Printing

```{eval-rst}

.. currentmodule:: progressivis.io.print

.. autoclass:: Every
   :members:
   :exclude-members: run_step, is_data_input, get_progress
```

```{eval-rst}

.. currentmodule:: progressivis.io.print

.. autoclass:: Print
   :members:
   :exclude-members: run_step, is_data_input, get_progress
```


```{eval-rst}

.. currentmodule:: progressivis.io.print

.. autoclass:: Tick
   :members:
   :exclude-members: run_step, is_data_input, get_progress
```


### Simple CSV Loader



```{eval-rst}

.. currentmodule:: progressivis.io.simple_csv_loader

.. autoclass:: SimpleCSVLoader
   :members:
   :exclude-members: run_step, is_data_input, get_progress
```

### Reliable CSV Loader

```{eval-rst}
.. currentmodule:: progressivis.io.csv_loader

.. autoclass:: CSVLoader
   :members:
   :exclude-members: run_step, is_data_input, get_progress, rows_read
```


### CSV Loader via PyArrow

```{eval-rst}
.. currentmodule:: progressivis.io.pa_csv_loader

.. autoclass:: PACSVLoader
   :members:
   :exclude-members: run_step
```


### Parquet Loader via PyArrow

```{eval-rst}
.. currentmodule:: progressivis.io.parquet_loader

.. autoclass:: ParquetLoader
   :members:
   :exclude-members: run_step
```
<!---
progressivis.io.input.Input
--->
### Constant Module

```{eval-rst}
.. currentmodule:: progressivis.table.constant

.. autoclass:: Constant
   :members:
   :exclude-members: run_step
```

```{eval-rst}
.. currentmodule:: progressivis.table.constant

.. autoclass:: ConstDict
   :members:
   :exclude-members: run_step
```



### Variable Module

```{eval-rst}
.. currentmodule:: progressivis.io.variable

.. autoclass:: Variable
   :members:
   :exclude-members: run_step, is_input, has_input
```


## Data Filtering Modules

### Simple Filtering Module


```{eval-rst}
.. currentmodule:: progressivis.table.simple_filter

.. autoclass:: SimpleFilter
   :members:
   :exclude-members: run_step, parameters
```

### CMP Query Module (via NumExpr)


```{eval-rst}
.. currentmodule:: progressivis.table.cmp_query

.. autoclass:: CmpQueryLast
   :members:
   :exclude-members: run_step
```

### Numerical Expression Filtering Module (via NumExpr)


```{eval-rst}
.. currentmodule:: progressivis.table.filtermod

.. autoclass:: FilterMod
   :members:
   :exclude-members: run_step, parameters
```
Example:

```{eval-rst}
.. literalinclude:: ./example_filtermod.py
```

The underlying graph:

```{eval-rst}
.. progressivis_dot:: ./example_filtermod.py
```



### The Range Query Module


```{eval-rst}
.. currentmodule:: progressivis.table.range_query

.. autoclass:: RangeQuery
   :members:
   :exclude-members: run_step, parameters
```
Module `RangeQuery` is not self-sufficient. It needs other modules to work. A simple way to provide it with an environment that allows it to work properly is to use the `create_dependent_modules()` method.

This convenience method creates a set of modules connected to `RangeQuery` that produce the inputs required for its operation in most cases.


In the example below it is called with the default values:

```{eval-rst}
.. literalinclude:: ./example_range_query.py
```

And in this case it produces the following topology:

```{eval-rst}
.. progressivis_dot:: ./example_range_query.py
```

### The Range Query 2D Module


```{eval-rst}
.. currentmodule:: progressivis.table.range_query_2d

.. autoclass:: RangeQuery2D
   :members:
   :exclude-members: run_step, parameters
```

Just like `RangeQuery`, the module `RangeQuery2D` is not self-sufficient. In order to provide it with an environment, the `create_dependent_modules()` method can be used in the same way:

```{eval-rst}
.. literalinclude:: ./example_range_query_2d.py
```

### Categorical Query Module


```{eval-rst}
.. currentmodule:: progressivis.table.categorical_query

.. autoclass:: CategoricalQuery
   :members:
   :exclude-members: run_step
```

Like previous query modules `CategoricalQuery` is not self-sufficient. Use the `create_dependent_modules()` to initialize its environement:

```{eval-rst}
.. literalinclude:: ./example_categorical_range.py
```

```{eval-rst}
.. progressivis_dot:: ./example_categorical_range.py
```
<!---
table.select.Select: Vue sur un PTable avec un select: bitmap
table.liteselect.LiteSelect: même chose avec une vue en sortie. devrait s'appeler selectview
--->

## Indexing Modules

### Binning Index Module

```{eval-rst}
.. currentmodule:: progressivis.table.binning_index

.. autoclass:: BinningIndex
   :members:
   :exclude-members: run_step, parameters
```

### Binning Index N-dim Module

```{eval-rst}
.. currentmodule:: progressivis.table.binning_index_nd

.. autoclass:: BinningIndexND
   :members:
   :exclude-members: run_step, parameters
```

### Unique Index Module

```{eval-rst}
.. currentmodule:: progressivis.table.unique_index

.. autoclass:: UniqueIndex
   :members:
   :exclude-members: run_step
```

## Data Grouping/Joining/Aggregation Modules

### Join-by-id Module

```{eval-rst}
.. currentmodule:: progressivis.table.join_by_id

.. autoclass:: JoinById
   :members:
   :exclude-members: run_step
```

### Join Module

```{eval-rst}
.. currentmodule:: progressivis.table.join

.. autoclass:: Join
   :members:
   :exclude-members: run_step
```

```{eval-rst}
.. literalinclude:: ./example_join.py
```

```{eval-rst}
.. progressivis_dot:: ./example_join.py
```



### Group-By Module

```{eval-rst}
.. currentmodule:: progressivis.table.group_by

.. autoclass:: GroupBy
   :members:
   :exclude-members: run_step
```

### Aggregate Module

```{eval-rst}
.. currentmodule:: progressivis.table.aggregate

.. autoclass:: Aggregate
   :members:
   :exclude-members: run_step
```

## Set and Flow Control Operations


### Intersection Module

```{eval-rst}
.. currentmodule:: progressivis.table.intersection

.. autoclass:: Intersection
   :members:
   :exclude-members: run_step, run_step_progress, run_step_seq
```

### Merging dictionaries Module

```{eval-rst}
.. currentmodule:: progressivis.table.merge_dict

.. autoclass:: MergeDict
   :members:
   :exclude-members: run_step
```

### Switching Module

```{eval-rst}
.. currentmodule:: progressivis.table.switch

.. autoclass:: Switch
   :members:
   :exclude-members: run_step
```



<!---
table.pattern.Pattern:
table.paste.Paste:
table.merge.Merge:
--->



## Statistical Modules

### Histograms

```{eval-rst}
.. currentmodule:: progressivis.stats.histogram1d

.. autoclass:: Histogram1D
   :members:
   :exclude-members: run_step, is_ready, reset, parameters

.. currentmodule:: progressivis.stats.histogram2d

.. autoclass:: Histogram2D
   :members:
   :exclude-members: run_step, is_ready, reset, parameters

.. currentmodule:: progressivis.stats.histogram1d_categorical

.. autoclass:: Histogram1DCategorical
   :members:
   :exclude-members: run_step, is_ready, reset, parameters
```


### Max / IdxMax

```{eval-rst}
.. currentmodule:: progressivis.stats.max

.. autoclass:: Max
   :members:
   :exclude-members: run_step, is_ready, reset

.. currentmodule:: progressivis.stats.idxmax

.. autoclass:: IdxMax
   :members:
   :exclude-members: run_step, is_ready, reset, parameters
```

### Min / IdxMin

```{eval-rst}
.. currentmodule:: progressivis.stats.min

.. autoclass:: Min
   :members:
   :exclude-members: run_step, is_ready, reset

.. currentmodule:: progressivis.stats.idxmin

.. autoclass:: IdxMin
   :members:
   :exclude-members: run_step, is_ready, reset, parameters
```

### Random sampling

```{eval-rst}
.. currentmodule:: progressivis.stats.random_table

.. autoclass:: RandomPTable
   :members:
   :exclude-members: run_step, is_ready, reset, parameters
```

```{eval-rst}
.. currentmodule:: progressivis.stats.blobs_table

.. autoclass:: BlobsPTable
   :members:
   :exclude-members: run_step, is_ready, reset, parameters, kw_fun

.. autoclass:: MVBlobsPTable
   :members:
   :exclude-members: run_step, is_ready, reset, parameters, kw_fun
```


### Stats

```{eval-rst}
.. currentmodule:: progressivis.stats.stats

.. autoclass:: Stats
   :members:
   :exclude-members: run_step, is_ready, reset, parameters
```


### Sample

```{eval-rst}
.. currentmodule:: progressivis.stats.sample

.. autoclass:: Sample
   :members:
   :exclude-members: run_step, parameters
```

### Variance

```{eval-rst}
.. currentmodule:: progressivis.stats.var

.. autoclass:: Var
   :members:
   :exclude-members: run_step, parameters

.. autoclass:: VarH
   :members:
   :exclude-members: run_step, parameters
```



<!---
stats.mchistogram2d.MCHistogram2D:
stats.scaling.MinMaxScaler:
stats.max.ScalarMax:

stats.cxxmax.Max:
stats.kernel_density.KernelDensity:
stats.distinct.Distinct:
stats.correlation.Corr:
stats.counter.Counter:
stats.percentiles.Percentiles:
stats.kll.KLLSketch:
stats.min.ScalarMin:
stats.ppca.PPCA:
stats.ppca.PPCATransformer:
table.percentiles.Percentiles:
--->

## Clustering Modules

<!---
cluster.mb_k_means.MBKMeans:
cluster.mb_k_means.MBKMeansFilter:

--->


## Linear Algebra Modules

<!---
# To generate csv files containing the lists of unary/binary modules use:
python -c 'from progressivis.linalg.elementwise import generate_unary_csv as f;f("doc/linalg_unary.csv")'
python -c 'from progressivis.linalg.elementwise import generate_binary_csv as f;f("doc/linalg_binary.csv")'
--->


### Element-wise processing modules

```{eval-rst}
These modules apply :term:`numpy universal functions (a.k.a. ufunc) <ufunc>` to the column of one or two input tables.

Depending on the applied :term:`ufunc` arity, we distinguish several categories of modules.
```
#### Unary modules

```{eval-rst}
These modules apply an unary :term:`ufunc` on all columns or a subset of columns from the input table, for example:

.. currentmodule:: progressivis.linalg

.. autoclass:: Absolute
   :members:
```

The other unary modules have the same interface as ``Absolute`` module above. They are:

```{eval-rst}
.. csv-table:: Unary modules
   :file: linalg_unary.csv
   :widths: 30, 70
   :header-rows: 1
```

#### Binary modules

```{eval-rst}
These modules apply a binary :term:`ufunc` on two sets of columns belonging to same input table (for example ``ColsAdd`` below) or to two distinct tables (for example ``Add`` below):

.. currentmodule:: progressivis.linalg

.. autoclass:: Add
   :members:
```
Examples:

```{eval-rst}
.. literalinclude:: ./example_linalg_add.py
```


```{eval-rst}
.. currentmodule:: progressivis.linalg

.. autoclass:: ColsAdd
   :members:
```

Example:

```{eval-rst}
.. literalinclude:: ./example_linalg_cols_add.py
```


The other binary modules have the same interface as ``Add`` and ``ColsAdd`` modules above. They are:

```{eval-rst}
.. csv-table:: Binary and reduce modules
   :file: linalg_binary.csv
   :widths: 50, 50
   :header-rows: 1
```

#### Reduce modules

```{eval-rst}
These modules reduce input table columns dimensions by one, by applying an :term:`ufunc`, for example:

.. currentmodule:: progressivis.linalg

.. autoclass:: AddReduce
   :members:
```
The other reduce modules have the same interface as ``AddReduce`` modules above. They are listed in the previous table.


#### Decorators

One can transform a simple function into an element-wise module via three decorators ``@unary_module`` `@binary_module` and `@reduce_module`:

```{eval-rst}
.. currentmodule:: progressivis.linalg

.. autodecorator:: unary_module

.. autodecorator:: binary_module

.. autodecorator:: reduce_module
```

Examples:

```{eval-rst}
.. literalinclude:: ./example_linalg_decorators.py
```

#### Declarative modules

One can create new modules in a declarative way as subclasses of ``linalg.mixufunc.MixUfuncABC``.

Examples:

```{eval-rst}
.. literalinclude:: ./example_mix_ufunc.py
```
Declarative module based on ``numexpr`` expressions can be created using ``NumExprABC`` this way:

```{eval-rst}
.. literalinclude:: ./example_mix_nexpr.py
```

#### Linear mapping

Linear transformation can be performed via this module:

```{eval-rst}
.. currentmodule:: progressivis.linalg.linear_map

.. autoclass:: LinearMap
   :members:
   :exclude-members: run_step, parameters
```

Example:

```{eval-rst}
.. literalinclude:: ./example_linear_map.py
```

## Visualisation

<!---
vis.heatmap.Heatmap:
vis.stats_factory.DataShape:
vis.stats_factory.StatsFactory:
vis.histograms.Histograms:
vis.stats_extender.StatsExtender:
vis.mcscatterplot.MCScatterPlot:
--->

## Utility

<!---
core.wait.Wait:
core.module.Every:
core.sink.Sink:
table.wait_for_data.WaitForData:
table.stirrer.Stirrer:
table.stirrer.StirrerView:

--->

## Format Adaptors

<!---
table.repeater.Repeater: Vue sur PTable pour ajouter des cols calculés
table.reduce.Reduce: ???
table.dict2table.Dict2PTable:
table.last_row.LastRow:
table.combine_first.CombineFirst: paste table ???
--->
