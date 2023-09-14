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
   :exclude-members: run_step
```


## Data Filtering Modules

### Bisect Query Module (via NumExpr)


```{eval-rst}
.. currentmodule:: progressivis.table.bisectmod

.. autoclass:: Bisect
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

## Statistical Modules

### Sample

```{eval-rst}
.. currentmodule:: progressivis.stats.sample

.. autoclass:: Sample
   :members:
   :exclude-members: run_step
```

## Linear Algebra Modules
