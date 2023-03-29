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

.. autoclass:: PTable
   :members:

.. autoclass:: PTableSelectedView
   :members:
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

