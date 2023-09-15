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

Under the hood, a ProgressiVis program is a dataflow graph, stored in a Dataflow object.
When creating or updating a program, a dataflow graph is updated and validated before it is run by the scheduler. There is a two-phase-commit cycle to make sure the currently running dataflow is not broken by an invalid modification.

Once the dataflow is valid, it is run by the scheduler in a round-robin fashion.

### Scheduler


```{eval-rst}
.. currentmodule:: progressivis.core

.. autoclass:: Scheduler
   :members:
   :exclude-members: __init__, new_run_number, start_impl, run, or_default, for_input, fix_quantum, time_left
```

### Dataflow

To create or update a program, you need to get a `Dataflow` object. A scheduler is a context manager returning a `Dataflow`. Use it like this:
```Python
scheduler = Scheduler.default
with scheduler as dataflow:
    m = Max(name="max", scheduler=scheduler)
    prt = Print(name="print_max", proc=proc, scheduler=scheduler)
    m.input.table = table.output.result
    prt.input.df = m.output.result
    dataflow.delete_modules("min", "print_min")
```
This example creates and add two new modules (`max` and `print_max`) to the current `Dataflow` and removes two other modules (`min` and `print_min`).

The context manager can succeed, updating the scheduler with the new dataflow, or fail, producing an error and not updating the scheduler.  In that case, the exception contains a structured error message explaining in human terms the problems.  Alternatively, the `Dataflow.validate()` method returns a list of errors (possibly empty) in the current `Dataflow` that can be fixed before the context manager fails.

```{eval-rst}
.. currentmodule:: progressivis.core.dataflow

.. autoclass:: Dataflow
   :members:
   :exclude-members: __init__, generate_name, validate_module_inputs, validate_module_outputs, validate_module
```

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

