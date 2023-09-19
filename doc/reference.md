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
    prt = Print(name="print_max", scheduler=scheduler)
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

A `Module` is used as a function in a regular language. It provides a set of functionalities:
- connection,
- validation,
- execution,
- control,
- naming, tagging, and
- interaction.

The `Module` class is an abstract base class and cannot be instantiated. All the concrete modules inherits from this class.

```{eval-rst}
.. currentmodule:: progressivis.core.module

.. autoclass:: Module
   :members:
   :exclude-members: __new__, __init__, create_slot, connect_output, prepare_run, cleanup_run, ending, pretty_typename, start, terminate

```

#### Connection and Validation

When declaring a new module, its input slots, output slots, and parameters can be declared using three decorators: `@def_input`, `@def_output`, and `@def_parameter`.

```{eval-rst}
.. currentmodule:: progressivis.core.module

.. autodecorator:: def_input

.. autodecorator:: def_output

.. autodecorator:: def_parameter
```


For example, a new module can be declared like this:
```{code-block}
@def_parameter("history", np.dtype(int), 3)
@def_input("filenames", PTable, required=False, doc=FILENAMES_DOC)
@def_output("result", PTable, doc=RESULT_DOC)
class CSVLoader(Module):
  ...
```


Once a module is created, it should be connected to other modules. As shown in a previous example,
the syntax is:
```{code-block}
:linenos:
m = Max(name="max", scheduler=scheduler)
prt = Print(name="print_max", scheduler=scheduler)
m.input.table = table.output.result
prt.input.df = m.output.result
```

Here, on line 3, the input slot called `table` from the module `m` (the `max` module) is
connected to the output slot called `result` from the module `table` (its creation is not shown in the example). On line 4, the input slot `df` of `prt` (the `print_max` module) is connected to the output slot `result` of the `m` module.

Slot names are checked at creation time: you cannot refer to the name
a slot that is not declared in the module.  Slots are typed, and the
types are checked at validation time. Also some slots are required and
others are optional. Their existence is also checked at validation
time.

#### Name, Groups and Tags

Each module has a unique name, belongs to one group, and can have multiple tags associated with it. A name, group name, and tag name are simply strings.

At creation, a module is given a name, either explicitly if provided in the constructor (as in the example above, with names `max` and `print_max`), or automatically if not provided. This name is guaranteed to remain unique in a scheduler. If a name provided at creation time is already used, the creation throws an exception and the module is not created.

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

### Execution

TODO

#### Control

Callbacks:
```Python
ModuleProc = Callable[["Module", int], None]
```

`on_start_run`, `on_after_run`, `on_ending`


TODO



### Connections (Slots)

Internally, the connection between and input slot and an output slot is materialized with a `Slot` object. Each slot is typed to make sure an input slot is compatible with the output slots it connects to. At run time, slots carry data with the specified type.  There are four main types of progressive data that can be used in slots, as described in [](#progressive-data-structures).

Slots are also used by modules to know what changed in the data structures since the last time they were run, using the three `ChangeBuffer` attributes `created`, `updated`, and `deleted`.  Change management is described in detail in the next sections.

```{eval-rst}
.. currentmodule:: progressivis.core.slot

.. autoclass:: Slot
   :members:
   :exclude-members: __init__, __str__, __repr__, __eq__, __neq__, connect
```


### Change Management

Modules are run multiple times for a limited amount of time to perform their computation.  In between two runs, their input data can be changed by other modules upwards in the dataflow.
Under the hood, `ProgressiVis` tracks what changes from one run to the next in the progressive data structures carried by slots.  To simplify this change management, each data structure is considered as an indexed collection, with indices ranging from 0 to the length of the data structure. It means that these data structures are indexed with a main axis.  The information regarding the changes are limited to new `created` entries at specified indices, `deleted` entries at specified indices, and `updated` entries at specified indices. For the updated entries, the old values are not kept (so far).

Therefore, when a module runs, it can access its input data through the input slots, and can see what has changed since the last run. It can then decide to act according to the changes.  For details, see the [Custom Modules](./custom_modules.md#custom-modules) section.

## Progressive Data Structures

ProgressiVis relies mainly on four specially designed data types:

* tables/views
* columns
* dictionaries
* bitmaps.

These data types are instrumented to keep track of the changes happening between two consecutive module runs.

### Interaction

### Tables and views


One of the most important progressive data structure in `ProgressiVis` is the `Table`. It is similar to Pandas `DataFrame`, with several differences though, sometimes due to the progressive nature of the table, sometimes by design, and sometimes by lack of time to provide an interface as extensive as pandas.

Contrary to a pandas DataFrame, a `PTable` can grow and shrink efficiently. For progressive operations such as progressively loading a table from a file and computing derived values from a table progressively loaded.

Tables come into two main concrete classes: `PTable` and `PTableSelectedView`.

`ProgressiVis` has been designed with user interaction in mind. It supports efficient dynamic filtering of large data tables to restrict the visualization or analyses to subsets of the loaded data. A `PTableSelectedView` is simply a `PTable` filtered by a `PIntSet`.  It is used in many places, either for user filtering or for synchronizing multiple data structures to the same set of valid indices.

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
#### Columns

A table is a collection of named and typed columns.  In a table, all the columns have the same length and all the elements in a column have the same type.

```{eval-rst}
.. currentmodule:: progressivis.table

.. autoclass:: BasePColumn
   :inherited-members:
   :exclude-members: __init__

.. autoclass:: PColumn
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

Progressive dictionaries are used as dictionaries with their changes tracked.

```{eval-rst}
.. currentmodule:: progressivis.utils.psdict

.. autoclass:: PDict
   :members:
```

### The PIntSet class

A PIntSet is a set of integer values implemented efficiently. They are used in many places in `ProgressiVis`, e.g., for indexing tables and for representing masks inside a `PTableSelectedView`.

```{eval-rst}
.. currentmodule:: progressivis.core.pintset

.. autoclass:: PIntSet
   :members:
```

## Decorators

### Method Decorators


### Class Decorators

