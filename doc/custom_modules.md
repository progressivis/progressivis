
# Custom Modules

New modules can be programmed in Python. They require some understanding of the internals of ProgressiVis. We introduce the main mechanisms step by step here.

To summarize, a module has a simple life cycle. It is first created and then connected to other modules in a dataflow graph. At some later point, it is validated by the scheduler. If something is wrong, it is not installed in the dataflow of the scheduler since the program created is invalid in some way and should be fixed by the user. For example, an input slot is connected to an incompatible output slot somewhere in the dataflow or a mandatory slot is not connected.

Once the dataflow graph is validated, the module is runnable but **blocked**. The scheduler will decide at some point to try to unblock and run it (explained later). When the module is **run**, its method `run_step()` is called with a few parameters; this is where the execution takes place.

## The `run_step()` method

The method `run_step()` needs to perform many operations to get its data from the input slots, know how long it should run, post data on its output slots, and report its progression. ProgressiVis provides several Python mechanisms and decorators to avoid typing long boilerplate code. While they simplify the syntax, their role should be understood to control the execution of progressive modules correctly.

The `run_step()` method can decide whether to let the module continue running or to stop it. When a module continues to run, it can be **blocked** or **ready**. A blocked module needs some input data to continue, whereas a ready module can be rescheduled without further testing by the scheduler.

When a module has finished, it becomes **terminated**. For example, once a CSV input module has finished loading a CSV file, its state becomes **terminated** and can be removed from the list of runnable modules.  Internally, the module first becomes a **zombie** to let the scheduler cleanup its dependency before it becomes **terminated**, but that's a small technical point.

If a module has a non-recoverable runtime error, it becomes **invalid**. For now, this is equivalent to **terminated**, but some debugging facilities could revive it in the future.

Finally, modules that are interactive can be resurrected after they are terminated.

## Cooperative Scheduling

ProgressiVis modules rely on **cooperative scheduling**, contrary to modern operating systems that use **preemptive scheduling**. In the latter, the scheduler decides on its own to interrupt the execution of a process to start another one. This decision is based on the time spent in the process and other factors that are opaque to the user (but can be found deep down in the description of the scheduler).

Instead, ProgressiVis relies on each module to abide by a specified **quantum**.

TODO

### Synchronization of Modules

When multiple modules are computing values over the same table, they may become desynchronized, some may be lagging behind due different processing speed.
