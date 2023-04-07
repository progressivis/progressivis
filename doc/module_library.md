# Module Library

## I/O Modules

### Simple CSV Loader

```{eval-rst}
.. currentmodule:: progressivis.io.simple_csv_loader

.. autoclass:: SimpleCSVLoader
   :members:
```

### Reliable CSV Loader

```{eval-rst}
.. currentmodule:: progressivis.io.csv_loader

.. autoclass:: CSVLoader
   :members:
```


### PyArrow BaseLoader

```{eval-rst}
.. currentmodule:: progressivis.io.base_loader

.. autoclass:: BaseLoader
   :members:
```
### CSV Loader via PyArrow

```{eval-rst}
.. currentmodule:: progressivis.io.pa_csv_loader

.. autoclass:: PACSVLoader
   :members:
```


### Parquet Loader via PyArrow

```{eval-rst}
.. currentmodule:: progressivis.io.parquet_loader

.. autoclass:: ParquetLoader
   :members:
```


### Constant Module

```{eval-rst}
.. currentmodule:: progressivis.table.constant

.. autoclass:: Constant
   :members:
```


### ConstDict Module

```{eval-rst}
.. currentmodule:: progressivis.table.constant

.. autoclass:: ConstDict
   :members:
```



### Variable Module

```{eval-rst}
.. currentmodule:: progressivis.io.variable

.. autoclass:: Variable
   :members:
```


### Dynamic Variable Module

```{eval-rst}
.. currentmodule:: progressivis.io.dynvar

.. autoclass:: DynVar
   :members:
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

### The Range Query Module


```{eval-rst}
.. currentmodule:: progressivis.table.range_query

.. autoclass:: RangeQuery
   :members:
   :exclude-members: run_step
```

The modules topology produced by `create_dependent_modules()` is shown below:

```{eval-rst}
.. graphviz::

   digraph range_query_dg {
      subgraph tier1 {
        node [color="lightgreen",style="filled",group="tier1"]
        RangeQuery
      }


      "Input" -> "HistogramIndex";
      "HistogramIndex" -> "Min" [label=min_out];
      "HistogramIndex" -> "Max"[label=max_out];
      "Variable[lo]" -> "RangeQuery"  [label=lower];
      "Min" -> "Variable[lo]" [label=like];
      "HistogramIndex" -> "RangeQuery" [label=hist];
      "Max" -> "Variable[up]" [label=like];
      "Variable[up]" -> "RangeQuery"  [label=upper];
      "Min" -> "RangeQuery" [label=min];
      "Max" -> "RangeQuery" [label=max];
      "RangeQuery" -> "Output" [label=min];
      "RangeQuery" -> "Output" [label=result];
      "RangeQuery" -> "Output" [label=max];

   }

```

### The Range Query 2D Module


```{eval-rst}
.. currentmodule:: progressivis.table.range_query_2d

.. autoclass:: RangeQuery2d
   :members:
   :exclude-members: run_step
```
The modules topology produced by `create_dependent_modules()` is shown below:

```{mermaid}
flowchart TD
    hi(HistogramIndex/lo)
    rq(RangeQuery)
    in((Input))
    o((Output))
    in --> hi
    hi -->|min_out: min|rq
    hi -->|max_out: max|rq
    hi -->|result: table|rq
    rq -->|result|o

```



```{eval-rst}
.. graphviz::

   digraph range_query_dg {
      subgraph tier1 {
        node [color="lightgreen",style="filled",group="tier1"]
        RangeQuery
      }


      "Input" -> "HistogramIndex";
      "HistogramIndex" -> "Min" [label=min_out];
      "HistogramIndex" -> "Max"[label=max_out];
      "Variable[lo]" -> "RangeQuery"  [label=lower];
      "Min" -> "Variable[lo]" [label=like];
      "HistogramIndex" -> "RangeQuery" [label=hist];
      "Max" -> "Variable[up]" [label=like];
      "Variable[up]" -> "RangeQuery"  [label=upper];
      "Min" -> "RangeQuery" [label=min];
      "Max" -> "RangeQuery" [label=max];
      "RangeQuery" -> "Output" [label=min];
      "RangeQuery" -> "Output" [label=result];
      "RangeQuery" -> "Output" [label=max];

   }



```

### Categorical Query Module


```{eval-rst}
.. currentmodule:: progressivis.table.categorical_query

.. autoclass:: CategoricalQuery
   :members:
   :exclude-members: run_step
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
