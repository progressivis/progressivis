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

A **Graphviz** variant:

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

A simple **Mermaid** variant:

```{mermaid}
flowchart TD
    classDef green fill:#9f6,stroke:#333,stroke-width:2px;
    classDef orange fill:#f96,stroke:#333,stroke-width:4px;
    hi(HistogramIndex)
    rq(RangeQuery)
    in((Input))
    o((Output))
    in --> hi
    hi--min_out: min-->rq
    hi -->|max_out: max|rq
    hi -->|result: table|rq
    rq -->|result|o
    class rq green
```

Improved **Mermaid** variant:

```{mermaid}
  flowchart TD
    classDef outslot fill:#f96,stroke:#333,stroke-width:1px;
    result_hi(result)
    table_rq(table)
    table_hi(table)
    table_dyn_lo(table)
    table_dyn_up(table)
    res_dyn_lo(result)
    res_dyn_up(result)
    min_rq(min)
    max_rq(max)
    Input-->table_hi
    Input-->table_rq
    res_rq(result)-->Output
    min_out(min_out)-->table_dyn_lo
    max_out(max_out)-->table_dyn_up
    result_hi-->hist
    res_dyn_lo-->lower
    res_dyn_up-->upper
    min_out-->min_rq
    max_out-->max_rq
    evt_low>From input lo]-->DynVar/lo
    evt_up>From input up]-->DynVar/up
    subgraph DynVar/lo
     table_dyn_lo
     res_dyn_lo
    end
    subgraph DynVar/up
     table_dyn_up
     res_dyn_up
    end
   subgraph HistogramIndex
    direction TB
    subgraph Inputs
     table_hi
    end
    subgraph Outputs
     min_out
     max_out
     result_hi
    end
    Inputs-.-Outputs
   end
   subgraph RangeQuery
    direction TB
    subgraph inputs2 [Inputs]
     direction LR
     hist
     lower
     upper
     min_rq
     max_rq
     table_rq
    end
    subgraph outputs2 [Outputs]
     direction LR
     res_rq
    end
    inputs2 -.- outputs2
   end
  class result_hi,min_out,max_out,res_dyn_lo,res_dyn_up,res_rq outslot
```


### The Range Query 2D Module


```{eval-rst}
.. currentmodule:: progressivis.table.range_query_2d

.. autoclass:: RangeQuery2d
   :members:
   :exclude-members: run_step
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
