# Writing New Modules

New modules can be programmed in Python. They require some understanding of the internals of ProgressiVis. We introduce the main mechanisms step by step here.

To summarize, a module has a simple life cycle. It is first created and then connected to other modules in a dataflow graph. At some later point, the dataflow is [validated by the scheduler](#validity). If something is wrong, the new dataflow is not installed in the scheduler since the program is invalid in some way and should be fixed by the user. For example, a required input slot is not connected to the module.

Once the dataflow graph is validated, the module is runnable and its state turns to **ready**. When the module is **run**, its method `run_step()` is called with a few parameters; this is where the execution takes place.

As a simple example, the dataflow of the [user guide example](#quantiles-variant) is shown below:
```{eval-rst}
.. progressivis_dot:: ./userguide1.py
```
The modules are first ordered linearly using [topological sorting](https://en.wikipedia.org/wiki/Topological_sorting), according to the dataflow graph made of modules linked by slots. Then, the scheduler runs them in order and starts again in the end.
For each module, the scheduler calls the method `is_ready()` and, if it returns `True`, it calls the method `run()` that calls the method `run_step()` described in the next section.
When reaching the end of the module list (called the `run_list` in the Scheduler), the Scheduler cleans-up its list by removing all the modules that have finished their work.
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

The method `Module.run_step()` needs to perform several operations to get its data from the input slots, know how long it should run, post data on its output slots, and report its progression. ProgressiVis provides several Python mechanisms and decorators to avoid typing long boilerplate code. While they simplify the syntax, their role should be understood to control the execution of progressive modules correctly.

`Module.run_step` is called by `Module.run`, which is not meant to be redefined in subclasses of `Module`. `Module.run` prepares the arguments of `run_step` and collects its return values or exception to monitor the module execution.

At a high level, `Module.run_step` performs the following operations:
1. **Input Slot Management** See which input slots have changed and decide how much work it can do given its quantum; this work can become **chunks** of data to process, **number iterations** to perform, or both.
2. **Partial Computation** Run the internal computation.
3. **Preparing the Output** Fill the output slots with the approximate or partial results.
4. **Update State and Speed** Return information on its new state and the actual work that has been performed.

When a module has finished its computation, it becomes **terminated**.
For example, once a CSV loader module has finished loading a CSV file, its state becomes **terminated** and it can be removed by the scheduler from the list of runnable modules.
Internally, the module first becomes a **zombie** to let the scheduler clean up its dependent modules before it becomes **terminated**, but that's a small technical point.

If a module has a non-recoverable runtime error or raises an exception, it becomes **invalid**. For now, this is equivalent to **terminated**, but some debugging facilities could revive it in the future.

Finally, modules that are interactive can be resurrected after they are terminated. We address this point in the [interactive behavior section](#interactive_behavior).

Implementing a new module mostly boils down to implementing its `run_step()` method. An example is given [below](#max_module).

## Cooperative Scheduling

ProgressiVis modules rely on **cooperative scheduling**, contrary to modern operating systems that use **preemptive scheduling**. In the latter, the scheduler decides on its own to interrupt the execution of a process to start another one. This decision is based on the time spent in the process and other factors that are opaque to the user (but can be found deep down in the description of the scheduler).

Instead, ProgressiVis relies on each module to abide by a specified **quantum** of time.  It means that, when the main method `run_step()` of a module is called, it is given a quantum. Within this quantum, it should perform its computation, return a useful result (approximate or partial if needed), and return information regarding its state, either `ready`, `blocked`, or `zombie` (about to terminate but still alive).

ProgressiVis cannot use preemptive scheduling because of step 3; an arbitrary computation cannot be interrupted at any point and return a meaningful result. It should stop at a consistent point in its computation to prepare and provide a meaningful result.


(max_module)=
## Example: The SimpleMax Module

The `SimpleMax` module is a simplification of the `Max` module of ProgressiVis.
It computes the maximum values of all the columns of the table that it takes in an input slot and returns its result as a `PDict`, a dictionary that associates with each column name its maximum value, according to the data already processed progressively.
Let's explain all its parts step by step.

```{eval-rst}
.. literalinclude:: ./simple_max.py
   :linenos:
```

### Input/Output/Parameters Definition

ProgressiVis defines several Python decorators to limit the amount of boilerplate code to type.
Every Module class uses some input slots, output slots, and parameters. They can be declared using the `@def_input`, `@def_output`, and `@def_paramameter` decorators. These decorators can appear after the `@document` decorator (line 10).

Line 11 declares an input slot called "table" of type `PTable` and provides a short documentation.

Line 12 declares the output slot called "result", of type `PDict`, i.e., a "progressive dictionary".
It will contain the maximum value of each column computed progressively. The output slot descriptor also defines a documentation string.

Input and output slots can also be required or not; by default, they are required. When a slot is required, it should be connected for a dataflow configuration to be **valid**. We discuss the notion of dataflow validity [next](#validity).

Line 14 defines the class `SimpleMax`, inheriting from the `Module` class.
Its `__init__` method is very standard and just catches the keyword parameter passed to keep it as an instance variable.
It is redefined to initialize the value of the `default_step_size` instance variable with a reasonable value for the `SimpleMax` module, as explained in the next [Time Predictor section](#time-predictor).
Without the `@def_` decorators, the `__init__` method would require many more lines of code to  declare the slots and parameters.

(validity)=
### Validity of a Dataflow

To run, a dataflow should be **valid**. The validity is defined as follows:
- For all the modules, all the required slots should be connected
- For all the connected slots, the input and output slots should be compatible
- There should not be any cycle in the dataflow; it should be a **directed acyclic graph**

By design, ProgressiVis checks the connection types as soon as they are specified. However, when building or modifying a dataflow graph, adding modules or removing modules, the dataflow graph remains invalid until all the connections are made and dependent modules are deleted from the dataflow. Therefore, checking for the required slots and cycles is done as a two-phase commit operation.


### The `SimpleMax.run_step` Method
The method that performs the main work of a module is:\
`run_step(self, run_number: int, step_size: int, quantum: float) -> ReturnRunStep`.
It takes three arguments. The first, `run_number`, is an integer provided by the Scheduler. Each time the scheduler calls the `run` method of a module, it increments that number. The `run_number` is a convenient timestamp, typically used to mark an operation performed on a data structure, e.g., to check if something has changed since the last run of `run_step()`.

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
        input_module: Optional[Module]
        input_name: Optional[str]
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
Line 24 checks if any item in the table has been updated or deleted since the last call to `next_step`. This information is obtained because ProgressiVis's data structures are designed to keep track of these changes made by modules.

(change_management)=
### Managing Changes


ProgressiVis calls `run_step` iteratively on all the modules. When entering the method, it is necessary to know what has changed since the last call. All the progressive data structures of ProgressiVis provide this information through an internal mechanism. The slot holding a `PTable` can be queried to know the table lines that have been created, updated, and deleted. The mechanism is identical for a slot containing a `PDict`; each key is given an index so the change mechanism returns the number of keys created, updated, and deleted. `PIntSet` also keeps track of its changes.

In our example, we only deal with created items. If the table had been changed by removing items or updating the value of items, line 25 resets the slot. The slot starts anew, ignoring all the previous operations. The `Slot.update(int)` method will then update the slot, considering all the items in the table as created.

Once the slot has been reset, the value of the result dictionary should also be updated to minus infinity to be recomputed correctly in the next step.

This management of updated and deleted items is the simplest strategy to handle changes. It simply restarts the computation to the whole table. In many cases, better strategies are possible, but that one always works and can be used to start.

Note that the result `PDict`  should not be created again because of the change manager: the next modules rely on the key order to correctly handle changes. Creating another PDict would change the key order and break the change management.


### Partial Computation

Since the `SimpleMax` module only deals with one input slot, the "table", lines 29-31 extract the items of the table that have been created since the last call to `run_step()`. This is our **chunk** of data to process.  Note that the chunk extraction in line 31 does not copy values in the `PTable`, it creates a lightweight `PTableSelectedView`.

Line 33 computes the maximum value of all the columns of the chunk. The `PTable.max()` method performs this operation and returns the results in a dictionary.  This operation takes a time proportional to the size of the chunk.

### Preparing the Output

Lines 24-38 prepare the result of the module's partial execution. The result should be stored in the `self.result` instance variable. This is handled by the `@def_output` declaration at line 12.
The first time the module runs, the instance variable is `None`, so line 35 creates the `PDict` from the `op` dictionary. If the `PDict` is already created, it is updated key by key by applying the `numpy.fmax` function between the current value in the result `PDict` and the new value in the `op` variable.

### Update State and Speed

The `run_step()` method can decide whether to let the module continue running or to stop it. When a module continues to run, it can be **blocked** or **ready**. A blocked module needs some input data to continue, whereas a ready module can be rescheduled without further testing by the scheduler. This is checked by the method `is_ready()`; if the module state is `state_ready`, the module is ready to go, if it is `state_blocked`, it becomes ready when one of its input slots has more data available. Otherwise, the module is not ready.


## Difference with the `Max` Module

The `Max` module uses the standard `@process_slot` and `@run_if_any` decorators or `run_step` to shorten its code. It also implements **slot hints** in its "table" input slot.

The `hint_type` parameter specifies that this input slot can be parameterized using a sequence of strings. Concretely, all the connections made with slots of type "PTable" can be parameterized with a list of column names. We discuss these slot parameters in [slot hints](#slot_hints).

The code relies on two decorators for this function: `@process_slot` and `@run_if_any`.

`@process_slot` specifies that when the input slot "table" contains updated or deleted items (not created ones), the method `reset(self) -> None` should be called.  This method is defined on line 23.
The simplest strategy to use when an input table is modified is to restart the work from the beginning, forgetting the current "max" value.

`@run_if_any` means that if any of the input slots are modified, then the method should run. This is the default behavior for modules. This decorator also prepares the `self.context` context manager that does a great deal of bookkeeping.


(slot_hints)=
### Slot Hints

Slot hints provide a convenient syntax to adapt the behavior of slots according to parameters that we call "slot hints".
In PTable slots, the hints consist of a list of column names that restrict the columns received through the slot. Internally, this uses a PTable view. Creating a view can be done through a module, but the syntax is much heavier, and the performance is much worse.

In the [initial example](#quantiles-variant) of ProgressiVis, we use a `Quantiles` module where output slots can be parameterized by a quantile, such as 0.03 or 0.97.


## Synchronization of Modules

When multiple modules are computing values over the same table, they may become desynchronized; some may be lagging behind due to different processing speeds.

(interactive_behavior)=
## Interactive Behavior

TODO
