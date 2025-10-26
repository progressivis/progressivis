# Writing New Modules

New modules can be programmed in Python. They require some understanding of the internals of ProgressiVis. We introduce the main mechanisms step by step here.

## Overview

To summarize, a module has a simple life cycle. It is first created and then connected to other modules in a dataflow graph. At some later point, the dataflow is [validated by the scheduler](#validity). If something is wrong, the new dataflow is not installed in the scheduler since the program is invalid in some way and should be fixed by the user. For example, a required input slot is not connected to the module.

Once the dataflow graph is validated, the module is runnable and its state turns to **ready**. When the module is **run**, its method `run_step()` is called with a few parameters; this is where the main progressive execution takes place.

As a simple example, the dataflow of the [user guide example](#quantiles-variant) is shown below. For each module, the input slots are on the top, the output slots on the bottom, and the name (unique identifier) and type in the middle.
```{eval-rst}
.. progressivis_dot:: ./userguide1.py
```
The scheduler first orders the modules linearly using [topological sorting](https://en.wikipedia.org/wiki/Topological_sorting), according to the dataflow graph made of modules linked by slots. Then, the scheduler runs them in order and starts again at the end.
For each module, the scheduler calls the method `Module.is_ready()` and, if it returns `True`, it calls the method `Module.run()` that calls the method `Module.run_step()` described in the next section.
When reaching the end of the module list, the Scheduler cleans up its list by removing all the modules that have finished their work.
This means that modules have an internal state, `Module.state`, with the following possible values:
```Python
state_created
state_ready
state_running
state_blocked
state_suspended
state_zombie
state_terminated
state_invalid
```

Users only see these states in the process list, e.g., by looking at the value of the scheduler in the notebook. Programmers need to deal with at least `state_ready` and `state_blocked` when programming a new module.

## The `Module.run_step()` method

`Module.run_step` is called by `Module.run`, which is not meant to be redefined in subclasses of `Module`. `Module.run` is called by the scheduler and wraps `run_step`. It  prepares its arguments, calls it, and collects the return values or the exception to monitor the module execution.

The method `Module.run_step()` needs to perform several operations to get its data from the input slots, know how long it should run, run for its time quantum, post data on its output slots, and report its progression. ProgressiVis provides several Python mechanisms and decorators to avoid typing long boilerplate code. While they simplify the syntax, their role should be understood to control the execution of progressive modules correctly.

To summarize, at a high level, `Module.run_step` performs the following operations:
1. **Input Slot Management** See which input slots have changed and decide how much work it can do given its quantum; this work can become **chunks** of data to process, **number iterations** to perform, or both.
2. **Partial Computation** Run the internal computation.
3. **Preparing the Output** Fill the output slots with the approximate or partial results.
4. **Updating State and Speed** Return information on its new state and the actual work that has been performed.

When a module has finished its computation, it becomes **terminated**.
For example, once a CSV loader module has finished loading a CSV file, its state becomes **terminated** and it can be removed by the scheduler from the list of runnable modules.
Internally, the module first becomes a **zombie** to let the scheduler clean up its dependent modules before it becomes **terminated**, but that's a small technical point.

If a module has a non-recoverable runtime error or raises an exception, it becomes **invalid**. For now, this is equivalent to **terminated**, but some debugging facilities could revive it in the future.

Finally, modules that are interactive can be resurrected after they are terminated. We address this point in the [interactive behavior section](#interactive_behavior).

## Cooperative Scheduling

ProgressiVis's Scheduler implements **cooperative scheduling**, contrary to modern operating system schedulers that use **preemptive scheduling**. In the latter, the scheduler decides on its own to interrupt a process to start another one. This decision is based on the time spent in the process and other factors that are opaque to the user but try to be fair to all processes globally.

Instead, ProgressiVis's scheduler relies on each module to abide by a specified **quantum** of time.  It means that, when the method `Module.run_step()` is called, it is given a quantum. Within this quantum, it should perform its computation, return a useful result (approximate or partial if needed), and return information regarding its state, either `ready`, `blocked`, or `zombie` (about to terminate but still alive).

ProgressiVis cannot use preemptive scheduling because of step 3 above; an arbitrary computation cannot, in general, be interrupted at any point and return a meaningful result. It should stop at a consistent point in its computation to prepare and provide a meaningful result.


(max_module)=
## Example: The SimpleMax Module

The `SimpleMax` module is a simplification of the `Max` module of ProgressiVis.
It computes the maximum values of all the columns of the `PTable` that it takes in its input slot named "table" and returns its result in the output slot called "result" as a `PDict`, a dictionary that associates with each column name its maximum value. This running maximum value is updated according to the data already processed progressively.
Let's explain all the code parts step by step.

```{eval-rst}
.. literalinclude:: ./simple_max.py
   :linenos:
```

### Input/Output/Parameters Definition

ProgressiVis defines several Python decorators to limit the amount of boilerplate code to type.
Every Module class uses some input slots, output slots, and parameters. They can be declared using the `@def_input`, `@def_output`, and `@def_paramameter` decorators. These decorators appear at line 10-11.

Line 10 declares an input slot called "table" of type `PTable` and provides a short documentation.

Line 11 declares the output slot called "result", of type `PDict`, i.e., a "progressive dictionary".
The output slot descriptor also defines a documentation string.

Input and output slots can also be required or not; by default, they are required. When a slot is required, it should be connected for the dataflow to be **valid**. We discuss the notion of dataflow validity [in the next section](#validity).

Line 13 defines the class `SimpleMax`, inheriting from the `Module` class.
Its `__init__` method is minimal for modules; it catches all the keyword parameters to pass them to the `Module` constructor.
It is redefined only to initialize the value of the `default_step_size` instance variable with a reasonable value for the `SimpleMax` module, as explained in the next [Time Predictor section](#time-predictor).
Without the `@def_` decorators, the `__init__` method would require many more lines of code to  declare the slots and parameters.

(validity)=
### Validity of a Dataflow

To run in the Scheduler, a dataflow should be **valid**. The validity is defined as follows:
- For all the modules, all the required slots should be connected
- For all the connected slots, the input and output slots should be compatible
- There should not be any cycle in the dataflow; it should be a **directed acyclic graph**.

By design, ProgressiVis checks the connection types as soon as they are specified. However, when building or modifying a dataflow graph, adding modules or removing modules, the dataflow graph remains invalid until all the connections are made and dependent modules are deleted from the dataflow. Therefore, checking for the required slots and cycles is done as a two-phase commit operation.


### The `SimpleMax.run_step` Method
The method that performs the main work of a module is:\
`run_step(self, run_number: int, step_size: int, quantum: float) -> ReturnRunStep`.
It takes three arguments. The first, `run_number`, is an integer provided by the Scheduler. Each time it calls the `run` method of a module, it increments that number. The `run_number` is a convenient timestamp, typically used to mark an operation performed on a data structure, e.g., to check if something has changed since the last run of `run_step()`.

The last argument is simply the `quantum`, the maximum duration that the method is allowed to run. It is a floating point value specified in seconds (0.5 by default).

The `step_size` argument specifies how many **steps** the method should perform. In our example, it is the number of lines that it will handle from the input table, i.e., the **chunk size**. For other module classes, it can be the number of iterations to perform. The notion of **step** is interpreted by the module itself, but in many cases, the interpretation is the size of the chunk to process to stay within the quantum.

(time_predictor)=
### Time Predictor
ProgressiVis provides a mechanism to predict the number of steps the `run_step` method can perform within a given quantum: the **Time Predictor**.
Instead of only asking the `run_step` method to run for a given quantum of time, it also converts this quantum into `steps`.

It works as follows: the first time it runs a module, it uses the `default_step_size`, i.e., 10,000 lines (see line 17) in our example, and monitors the time needed to process that number of steps.
Assuming that run time is proportional to the number of lines processed, the time predictor computes a speed for the module (number of steps per second) and translates the quantum into a number of steps.

The time predictor adjusts this speed each time the module runs. It can accommodate a slight non-linearity, but it expects modules to spend a time roughly proportional to the number of steps, that is, to run each step at a constant speed.

### Input Slots Management

Lines 23-27 take care of the input slot. The `SimpleMax` module has only one input slot, "table", but other Module classes can have more. Most of the information needed to handle the input slot is accessible through the `Slot` object that implements the connection between modules. The `Slot` has the following interface (simplified):

```
@dataclass
class Slot:
        output_module: Module
        output_name: str
        input_module: Module
        input_name: str
        name: str
        hint: Any
        created: ChangeBuffer
        updated: ChangeBuffer
        deleted: ChangeBuffer
        def data() -> Any: ...
        def reset() -> None: ...
        def update(int) -> None: ...
```

Line 23 obtains the "table" input slot using the `Module.get_input_slot(name: str)->Slot` method.
Line 24 checks if any item in the table has been updated or deleted since the last call to `next_step`. This information is available because ProgressiVis's data structures are designed to keep track of these changes.

(change_management)=
### Managing Changes


ProgressiVis calls `run_step` iteratively on all the modules. When entering the method, it is necessary to know what has changed since the last call. All the progressive data structures of ProgressiVis provide this information through an internal mechanism. The slot holding a `PTable` can be queried to know the table lines that have been **created**, **updated**, and **deleted**. We call these three lists the change **Delta**.
The mechanism is identical for a slot containing a `PDict`; each key is given an index so the change mechanism returns the index of keys created, updated, and deleted. `PIntSet` also keeps track of its changes.

In our example, we only deal with created items. If the table had been changed by removing items or updating the value of items, line 25 resets the slot. The slot starts anew, ignoring all the previous operations. The `Slot.update(run_number: int)` method will then update the slot Delta, considering all the items in the table as created.

Once the slot has been reset, the value of the result dictionary should also be updated to minus infinity to be recomputed correctly in the next step.

This management of updated and deleted items is the simplest strategy to handle changes. It simply restarts the computation for the whole table. In many cases, better strategies are possible, but that one always works and can be used to start a Module implementation.

Note that the result `PDict`  should not be created again because of the change manager: the next modules rely on the key order to correctly handle changes. Creating another PDict would break the change management.


### Partial Computation

Since the `SimpleMax` module only deals with one input slot, the "table", lines 29-31 extract the items of the table that have been created since the last call to `run_step()`. This is our **chunk** of data to process.  Note that the chunk extraction in line 31 does not actually copy values in the `PTable`, it creates a lightweight `PTableSelectedView`, a filtered view of the `PTable`.

Line 33 computes the maximum value of all the columns of the chunk. The `PTable.max()` method performs this operation and returns the results in a dictionary.  This operation takes a time proportional to the size of the chunk.

### Preparing the Output

Lines 34-38 prepare the result of the module's partial execution. The result should be stored in the `self.result` instance variable. This is specified by the `@def_output` declaration at line 12.
The first time the module runs, the instance variable is `None`, so line 35 creates the `PDict` from the `op` dictionary. If the `PDict` is already created, it is updated key by key by applying the `numpy.fmax` function between the current value in the result `PDict` and the new value in the `op` variable.

### Updating State and Speed

The `run_step()` method can decide whether to let the module continue running or to stop it. When a module continues to run, it can be **blocked** or **ready**. A blocked module needs some input data to continue, whereas a ready module can be rescheduled without further testing by the scheduler. This is checked by the method `is_ready()`; if the module state is `state_ready`, the module is ready to go, if it is `state_blocked`, it becomes ready when one of its input slots has more data available. Otherwise, the module is not ready, and the scheduler will not try to run it.


## Difference with the `Max` Module

The `SimpleMax` module is very similar to the `Max` module; the latter additionally uses decorators for the `run_step` method, and manages **slot hints**.

The `Max` module uses the standard `@process_slot` and `@run_if_any` decorators of `run_step` to shorten its code.  These decorators save lines 25-28 of `SimpleMax` by introducing a `Context` in the management of the input slots.

```{eval-rst}
.. literalinclude:: ./max.py
   :linenos:
```

The `@process_slot` decorator in `Max` performs the equivalent of lines 25-28 of `SimpleMax`. It also creates an attribute in the `Context` to access the slot names "table", as shown in line 28.

`@process_slot` specifies that when the input slot "table" contains updated or deleted items (not created ones), the method `reset(self) -> None` should be called.  This method is defined on line 17.
The simplest strategy to use when an input table is modified is to restart the work from the beginning, forgetting the current "max" value.

The `@run_if_any` decorator specifies that the `run_step` method can be run when any of the input slots have new data. The `@run_if_all` method means that the `run_step` method can only run when all the input slots have new data. For most modules, the first is used.  A few modules that perform operations in parallel between their input slots, such as binary operators, need the latter.

The `hint_type` parameter specifies that this input slot can be parameterized using a sequence of strings. Concretely, all the connections made with slots of type "PTable" can be parameterized with a list of column names. We discuss these slot parameters in [slot hints](#slot_hints).


(slot_hints)=
### Slot Hints

Slot hints provide a convenient syntax to adapt the behavior of slots according to parameters.
In `PTable` slots, the hints consist of a list of column names that restrict the columns received through the slot. Internally, this uses a PTable view. Creating a view can be done through a module, but the syntax is much heavier, and the performance is much worse.


This slot hint is implemented in line 30 by calling `Module.filter_slot_columns`. The chunk returned will contain the columns specified in the slot hint, or all the columns if no slot hint is specified.

In the [initial example](#quantiles-variant) of ProgressiVis, we use a `Quantiles` module where output slots can be parameterized by a quantile, such as 0.03 or 0.97.


## Data Change Management

In the `SimpleMax` and `Max` examples, managing created items in a table is very efficient, but deleted or updated items trigger a complete recomputation through the `reset()` method.
Is there a better solution? In general, it is difficult to be definitive, but there are cases when a better answer is possible.

The `ScalarMax` module improves the `Max` module by keeping track of the items that reach the maximum value computed so far.  If, e.g., the indices `1, 10, 100` hold the maximum value, then deleting any other value does not invalidate the running maximum value. `ScalarMax` uses the `PIntSet` data structure to efficiently keep track of these indices. Yet, this management adds some overhead compared to the `Max` module when data is streamed in and never modified. Other implementations could even maintain more sophisticated data structures, trading efficiency depending on the expected frequency of the change events.

More work is needed to find other strategies to avoid resetting, but they will be specific to algorithms or classes of algorithms.


(interactive_behavior)=
## Interactive Behavior

ProgressiVis programs can be interactive; they can react to interactions using a mechanism based on the method `Module.from_input(msg: JSon)`.  It turns out that only one module class implements this mechanism in ProgressiVis, and can turn a static program into an interactive one: the `Variable`. For example, it can be used to provide the two parameters for a range filter, the minimum value and maximum value, as shown in [the user guide](#widgets_for_input). ProgressiVis provides such a filter that takes a table as input and outputs a filtered table. The filter can be implemented using an interactive range filter.

The `Variable.from_input()` method is typically called from a notebook widget, such as a range slider. The callback function creates a dictionary with keys and values that are sent to the `Variable.from_input()` method to specify, for example, the minimum value of the range filter, as a list of column names and values.

When the method is called, it copies the values in its output slot and notifies the scheduler that an interactive operation has been started. It calls the `Scheduler.from_input(mod: Module)` method. The scheduler changes its behavior to become interactive.

The interactive mode of the scheduler speeds up the activity of a part of the progressive dataflow. It first computes the subgraph between the input modules (the ones that called `Scheduler.from_input()` and the output modules, the ones that produce an output. Modules have properties, called `tags`, that are used to mark the "input", "source", and "visualization" modules, among others. Other tags can be added and removed from modules if needed by a program.  In interactive mode, the scheduler selects the subgraph between the input modules and the "visualization" modules reachable from them in the dataflow graph. This subgraph is then run in interactive mode for a short time (1.5 seconds) until it reverts to normal mode.  All the other modules are then run again, such as the data input modules.

When a program does not contain an input module, the scheduler will run it until all the modules are terminated. Modules blocked waiting for input from terminated modules are also terminated when they have consumed all the changed data from their input slots. Module termination propagates in a chain, and when all the modules are terminated, the scheduler stops.

When a program contains an input module, it means that the external world (a widget) can always send new data into the program. Therefore, the scheduler cannot terminate the input modules and their dependencies, and the program remains alive until the method `Scheduler.stop()` is called.

This mechanism is purely automatic; the only external control is based on the `Module.from_input()` method and is only implemented by the `Variable` module class so far, which has been sufficient to implement all the interactions needed.



## Synchronization of Modules

When multiple modules are computing values over the same table, they may become desynchronized; some may be lagging behind due to different processing speeds.



