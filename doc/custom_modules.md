
# Custom Modules

New modules can be programmed in Python. They require some understanding of the internals of ProgressiVis. We introduce the main mechanisms step by step here.

To summarize, a module has a simple life cycle. It is first created and then connected to other modules in a dataflow graph. At some later point, the dataflow is [validated by the scheduler](#validity). If something is wrong, the new dataflow is not installed in thescheduler since the program is invalid in some way and should be fixed by the user. For example, a required input slot is missing in a new module.

Once the dataflow graph is validated, the module is runnable but **blocked**. The scheduler will decide at some point to try to unblock and run it (explained later). When the module is **run**, its method `run_step()` is called with a few parameters; this is where the execution takes place.

As a simple example, dataflow of the [user guide example](#quantiles-variant) is shown below:
```{eval-rst}
.. progressivis_dot:: ./userguide1.py
```
The modules are first ordered linearly using [topological sorting](https://en.wikipedia.org/wiki/Topological_sorting). Then, the scheduler runs them in order and starts again in the end.
For each module, the scheduler calls the method `is_ready()` and, if it returns `True`, it calls the method `run()` that calls the method `run_step()` described in the next section.
When reaching the end of the module list (called the `run_list` in the Scheduler), the Scheduler cleans-up its list by removing all the modules that have finished their work.
This means that modules have an internal state, `Module.state`, with the following possible values:
```Python
class ModuleState(IntEnum):
    state_created = 0
    state_ready = 1
    state_running = 2
    state_blocked = 3
    state_suspended = 4
    state_zombie = 5
    state_terminated = 6
    state_invalid = 7
```

## The `run_step()` method

The method `run_step()` needs to perform many operations to get its data from the input slots, know how long it should run, post data on its output slots, and report its progression. ProgressiVis provides several Python mechanisms and decorators to avoid typing long boilerplate code. While they simplify the syntax, their role should be understood to control the execution of progressive modules correctly.

The `run_step()` method can decide whether to let the module continue running or to stop it. When a module continues to run, it can be **blocked** or **ready**. A blocked module needs some input data to continue, whereas a ready module can be rescheduled without further testing by the scheduler. This is checked by the method `is_ready()`; if the module state is `state_ready`, the module is ready to go, if it is `state_blocked`, it becomes ready if one of its input slots has more data available. Otherwise, the module is not ready.

When a module has finished, it becomes **terminated**. For example, once a CSV input module has finished loading a CSV file, its state becomes **terminated** and can be removed from the list of runnable modules.  Internally, the module first becomes a **zombie** to let the scheduler clean up its dependency before it becomes **terminated**, but that's a small technical point.

If a module has a non-recoverable runtime error, it becomes **invalid**. For now, this is equivalent to **terminated**, but some debugging facilities could revive it in the future.

Finally, modules that are interactive can be resurrected after they are terminated.

Implementing a new module mostly boils down to implementing its `run_step()` method. An example is given [below](#max_module).

## Cooperative Scheduling

ProgressiVis modules rely on **cooperative scheduling**, contrary to modern operating systems that use **preemptive scheduling**. In the latter, the scheduler decides on its own to interrupt the execution of a process to start another one. This decision is based on the time spent in the process and other factors that are opaque to the user (but can be found deep down in the description of the scheduler).

Instead, ProgressiVis relies on each module to abide by a specified **quantum** of time.  It means that, when the main method `run_step()` of a module is called, it is given a certain time. Within this time, it should perform its computation, return a useful result (approximate if needed), and return information regarding its state, either `ready`, `blocked`, or `zombie` (about to terminate but still alive).

Additionally, ProgressiVis provides an additional mechanism: the **Time Predictor**. Instead of only asking the `run_step` method to run for a given time, it also converts this time in `steps`.
Intuitively, when loading a given csv file, the time predictor reads a small number of lines and measures the time. Assuming that run time is linear with the number of lines read, the time predictor computes a throughput for the module and gives a number of steps the module should run to maintain its quantum.
The time predictor updates is throughput measure each time the module runs so it can accomodate a slight non-linearity but expect modules to spend a time proportional to the number of steps, or to run steps at a constant speed.

(change_management)=
## Managing Changes


(max_module)=
## Example: The Max Module

The `Max` module is among the simplest of ProgressiVis.
It computes the maximum values of all the columns of the table that it takes in an input slot.
Let's explain all its parts step by step.

```{eval-rst}
.. literalinclude:: ./max.py
   :linenos:
```

ProgressiVis defines several Python decorators to limit the amount of boilerplate code.
Every Module class uses input slots, output slots, and parameters. They can be declared using the `@def_input`, `@def_output`, and `@def_param` decorators. These decorators can appear after the `@document` decorator.

Line 12 declares an input slot called "table" of type `PTable`. The `hint_type` parameter specifies that this input slot can be parameterized using a sequence of strings. Concretely, all the connections made with slots of type "PTable" can be parameterized with a list of column names. We discuss these slot parameters in [slot hints](#slot_hints).

Line 13 declares the output slot called "result", of type `PDict`, i.e., a "progressive dictionary".
It will contain the maximum value of each column computed progressively. The output slot descriptor also defines a document string.

Input and output slots can also be required or not; by default, they are required. When a slot is required, it should be connected for a dataflow configuration to be **valid**. We discuss later the notion of dataflow validity [later](#validity).

Line 18 defined the class `Max`, inheriting from the `Module` class. Its `__init__` method is very standard and just catches the keyword parameter passed to keep it as an instance variable.
It is redefined to initialize the value of the `default_step_size` instance variable with a reasonable value for the `Max` module, as described in the [Time Predictor section](#time-predictor).
Without the `@def_` decorators, the `__init__` method would require many lines of code to  declare the slots and parameters.

The method that performs the main work of a module is `run_step(self, run_number: int, step_size: int, howlong: float) -> ReturnRunStep`.
The code relies on two decorators for this function: `@process_slot` and `@run_if_any`.

`@process_slot` specifies that when the input slot "table" contains updated or deleted items (not created ones), the method `reset(self) -> None` should be called.  This method is defined on line 23.
The simplest strategy to use when an input table is modified is to restart the work from the beginning, forgetting the current "max" value.

`@run_if_any` means that if any of the input slots are modified, then the method should run. This is the default behavior for modules. This decorator also prepares the `self.context` context manager that does a great deal of bookkeeping.

Since the `Max` module only deals with one input slot, the "table", the following lines extract the items of the table that have been modified since the last call to `run_step()`. These lines rely on the change management provided by ProgressiVis's progressive data structures.



(slot_hints)=
### Slot Hints

Slot hints provide a convenient syntax to adapt the behavior of slots according to parameters that we call "slot hints".
In PTable slots, the hints consist of a list of column names that restrict the columns received through the slot. Internally, this uses a PTable view. Creating a view can be done through a module, but the syntax is much heavier, and the performance is much worse.

In the [initial example](#quantiles-variant) of ProgressiVis, we use a `Quantiles` module where output slots can be parameterized by a quantile, such as 0.03 or 0.97.

(validity)=
### Validity of a Dataflow

To run, a dataflow should be **valid**. The validity is defined as follows:
- For all the modules, all the required slots should be connected
- For all the connected slots, the input and output slots should be compatible
- There should not be any cycle in the dataflow; it should be a **directed acyclic graph**

By design, ProgressiVis checks the connection types as soon as they are specified. However, when building or modifying a dataflow graph, adding modules or removing modules, the dataflow graph remains invalid until all the connections are made and dependent modules are deleted from the dataflow. Therefore, checking for the required slots and cycles is done as a two-phase commit operation.

(time_predictor) =
### Time Predictor



### Synchronization of Modules

When multiple modules are computing values over the same table, they may become desynchronized, some may be lagging behind due different processing speed.
