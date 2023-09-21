# Module Library

## I/O Modules

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
   :exclude-members: run_step
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
   :exclude-members: run_step
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
   :exclude-members: run_step
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

.. autoclass:: RangeQuery2d
   :members:
   :exclude-members: run_step
```

Just like `RangeQuery`, the module `RangeQuery2d` is not self-sufficient. In order to provide it with an environment, the `create_dependent_modules()` method can be used in the same way:

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

### Histogram Index Module

```{eval-rst}
.. currentmodule:: progressivis.table.hist_index

.. autoclass:: HistogramIndex
   :members:
   :exclude-members: run_step
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

<!---
table.intersection.Intersection:
table.hub.Hub: ...
table.pattern.Pattern:
table.merge_dict.MergeDict:
table.paste.Paste:
table.switch.Switch:
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

### Element-wise processing modules

```{eval-rst}
These modules apply :term:`numpy universal functions (a.k.a. ufunc) <ufunc>` to all columns or a subset of columns from the input table.

Depending on the applied :term:`ufunc` arity, we distinguish several categories of modules.
```
#### Unary modules

They apply an unary :term:`ufunc` this way:

```{eval-rst}
.. currentmodule:: progressivis.linalg

.. autoclass:: Absolute
   :members:

.. autoclass:: Arccos

.. autoclass:: Arccosh


```







<!---
linalg.mixufunc.MixUfuncABC:
linalg.elementwise.Unary:
linalg.elementwise.ColsBinary:
linalg.elementwise.Binary:
linalg.elementwise.Reduce:
linalg.nexpr.NumExprABC:
linalg.linear_map.LinearMap:
--->

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