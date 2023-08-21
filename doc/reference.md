# Library Reference

```{eval-rst}
.. currentmodule:: progressivis
```

## First level variables


```{eval-rst}
.. py:attribute:: __version__
```
    The version of the progressivis package.


## Dataflow Management

### Scheduler

### Module

### Connections (Slots)


### Change Management



### Interaction

## Progressive Data Structures

ProgressiVis relies mainly on four specially designed data types:

* tables/views
* columns
* dictionaries
* bitmaps.

These data types are instrumented to keep track of the changes happening between two consecutive module runs.

### Tables and views


Progressive tables:


```{eval-rst}
.. currentmodule:: progressivis.table

.. autoclass:: BasePTable
   :members:
   :exclude-members: __init__, index_to_id, id_to_index, info_contents, info_raw_contents, info_row

.. autoclass:: IndexPTable
   :members:
   :exclude-members: __init__

.. autoclass:: PTable
   :members:

.. autoclass:: PTableSelectedView
   :members:
```

#### Computed columns

In addition to stored columns, tables can contain virtual columns computed from the contents of other columns.
To create a computed column, you need to instantiate an object of class [](SingleColFunc), [](MultiColFunc) or [](MultiColExpr) and add it to the `computed` dictionary of the table. Subsequently, a view constructed on this table will be able to utilize the new column.


```{eval-rst}
.. currentmodule:: progressivis.table.compute

.. autoclass:: SingleColFunc
   :inherited-members:

.. autoclass:: MultiColFunc
   :inherited-members:

.. autoclass:: MultiColExpr
   :inherited-members:
```


### The PDict class

```{eval-rst}
.. currentmodule:: progressivis.utils.psdict

.. autoclass:: PDict
   :members:
```

### The PIntSet class


```{eval-rst}
.. currentmodule:: progressivis.core.pintset

.. autoclass:: PIntSet
   :members:
```

## Decorators

### Method Decorators


### Class Decorators

