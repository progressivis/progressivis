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

The `Scheduler` is in charge of running a ProgressiVis program, made of a sorted list of modules.


```{eval-rst}
.. currentmodule:: progressivis.core.scheduler

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

A Module is the equivalent of a function in a regular language. It provides a set of functionalities: connection, validation, execution, control, naming, tagging, and interaction.

The `Module` class is an abstract base class and cannot be instantiated. All the concrete modules inherits from this class.

```{eval-rst}
.. currentmodule:: progressivis.core.module

.. autoclass:: Module
   :members:
   :exclude-members: __new__, __init__, create_slot, connect_output, prepare_run, cleanup_run, ending, pretty_typename
```

#### Name, Groups and Tags

Each module has a unique name, belongs to one group, and can have multiple tags associated with it. A name, group name, and tag name are simply strings.

At creation, a module is given a name, either explicitly if provided in the constructor, or automatically if not provided. This name is guaranteed to remain unique in a scheduler. If a name provided at creation time is already used, the creation throws an exception and the module is not created.

A group is also a string, associated with a module at creation time. It is used when several modules are created to work together and should terminate together. A group name can be specified at creation time either in the constructor or using the `Module.grouped` context manager to associate the group name of a specified module to a set of newly created modules:
```Python
scheduler = Scheduler.default
mymainmodule = ...
with mymainmodule.grouped():
    m = Max(name="max", scheduler=scheduler)
    prt = Print(name="print_max", proc=proc, scheduler=scheduler)
    ...
```

Tags are used to add a simple attribute to a module. Any string can be used as tag, but a few are reserved to specify particular aspects of a module: `VISUALIZATION`, `INPUT`, `SOURCE`, `GREEDY`, `DEPENDENT`.

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

