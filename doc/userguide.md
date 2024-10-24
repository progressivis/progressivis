# User Guide


**ProgressiVis** is a language and system implementing **progressive data analysis and visualization**.
**ProgressiVis** is designed so that it never blocks in functions during unbounded amount of time, such as loading a large file from the network until completion.
In a traditional computation system, you don't worry much about the time taken when calling a function; loading a large file over the network is the price to pay for having it loaded and starting computations over its contents. In a progressive system, meant to be used interactively, when a function takes too long to complete, the user waits, get bored, and her attention drops.  **ProgressiVis** is designed to avoid this attention drop.
If you are familiar with asynchronous programming or real time programming, you will be familiar with the need to follow strict disciplines to make sure a system is not blocking.  This discipline is implemented everywhere in **progressiVis** with a specific "progressive" semantics.

## Key concepts

**ProgressiVis** uses specific constructs to remain reactive and interactive all the time.
Let's start with a simple progressive program to introduce the concept.  Assume we want to find out what are the popular places to go in New York City.
We can download the New York Taxi dataset that contain all the taxi trips in 2015 and 2016, including the pickup and drop-off positions, looking for hot-spots.

With **ProgressiVis**, we don't need to wait for the file to be fully loaded to visualize it, we can do it on the go, as with this simple low-level ProgressiVis program:

```python
from progressivis import CSVLoader, Histogram2D, Quantiles, Heatmap

LARGE_TAXI_FILE = "https://www.aviz.fr/nyc-taxi/yellow_tripdata_2015-01.csv.bz2"
RESOLUTION=512

csv = CSVLoader(LARGE_TAXI_FILE,
                index_col=False,
                usecols=['pickup_longitude', 'pickup_latitude'])

quantiles = Quantiles()
quantiles.input.table = csv.output.result

histogram2d = Histogram2D('pickup_longitude', 'pickup_latitude',
                          xbins=RESOLUTION, ybins=RESOLUTION)
histogram2d.input.table = csv.output.result
histogram2d.input.min = quantiles.output.result[0.03]
histogram2d.input.max = quantiles.output.result[0.97]

heatmap = Heatmap()
heatmap.input.array = histogram2d.output.result

heatmap.display_notebook()
csv.scheduler().task_start()
```

The image of all the taxi pickup positions appears immediately  and get more detailed progressively, revealing the shape of Manhattan and the two New York City airports.
![](images/nyc1.png)

With a standard visualization system, or using Pandas from python, you would have to wait a few minutes to see the visualization due to the load time of the file.
**ProgressiVis** show the results in a few seconds, improving over time, irrespective to the file size and network speed.

### Noisy Data

ProgressiVis is designed for scalability and managing big data. Big data is never clean; the taxi dataset is no exception. The simplest program for visualizing it is the following:
```python
from progressivis import CSVLoader, Histogram2D, Min, Max, Heatmap

LARGE_TAXI_FILE = "https://www.aviz.fr/nyc-taxi/yellow_tripdata_2015-01.csv.bz2"
RESOLUTION=512


csv = CSVLoader(LARGE_TAXI_FILE,
                index_col=False,
                usecols=['pickup_longitude', 'pickup_latitude'])

min = Min()
min.input.table = csv.output.result

max = Max()
max.input.table = csv.output.result

histogram2d = Histogram2D('pickup_longitude', 'pickup_latitude',
                          xbins=RESOLUTION, ybins=RESOLUTION)
histogram2d.input.table = csv.output.result
histogram2d.input.min = min.output.result
histogram2d.input.max = max.output.result

heatmap = Heatmap()
heatmap.input.array = histogram2d.output.result
...
```

Instead of using the `Quantiles` module, it simply uses `Min` and `Max` to obtain the bounds of the pickup positions before computing the heatmap image.
It works as well, but the resulting image is unexpected:

![](images/userguide_1_bad.png)

This is due to taxis driving to Florida (bottom right) or other far away places and forgetting to stop their meters.
The `Quantiles` module allows getting rid of outliers that always exist in real data, that is always noisy.

Alternatively, you may know the boundaries of NYC and specify them:
```python
from progressivis import CSVLoader, Histogram2D, ConstDict, Heatmap, PDict
from dataclasses import dataclass


LARGE_TAXI_FILE = "https://www.aviz.fr/nyc-taxi/yellow_tripdata_2015-01.csv.bz2"
RESOLUTION=512

@dataclass
class Bounds:
    top: float = 40.92
    bottom: float = 40.49
    left: float = -74.27
    right: float = -73.68

bounds = Bounds()
col_x = "pickup_longitude"
col_y = "pickup_latitude"


csv = CSVLoader(LARGE_TAXI_FILE,
                index_col=False,
                usecols=[col_x, col_y])

min = ConstDict(PDict({col_x: bounds.left, col_y: bounds.bottom}))
max = ConstDict(PDict({col_x: bounds.right, col_y: bounds.top}))

histogram2d = Histogram2D(col_x, col_y,
                          xbins=RESOLUTION, ybins=RESOLUTION)
histogram2d.input.table = csv.output.result
histogram2d.input.min = min.output.result
histogram2d.input.max = max.output.result

heatmap = Heatmap()
heatmap.input.array = histogram2d.output.result
...
```

The result is then perfect, but you need to provide extra information, i.e., the boundaries of the image.

![](images/userguide_1_ok.png)

## Main Components

In **ProgressiVis**, a program is run by a `Scheduler`. Only one instance of `Scheduler` exists (except in tests).
A progressive program is internally represented as a dataflow of progressive modules (simply called **modules** in this documentation).
The dataflow is a directed network with no cycle (a directed acyclic graph or DAG).

A `Module` represents the equivalent of a function in a traditional python program.
It is made of input and output slots; one output slot of a module can be connected to several input slots of other modules.
Some input slots are **optional**, and others are **mandatory** for a module to run.
Furthermore, a slot is **typed** since it carries data between modules.
A module with no input slot is a **source module**, and a module with no output slot is a **sink module**.
```{eval-rst}
.. _hint-reference-label:
```
Input slots can be supplemented by `hints`, provided in square brackets when specifying a connection.
The role and type of hints depends on the semantics of the slot. In the next example the sequence of names provided in square brackets designates the columns to be taken into account (and processed) by the module:

```python
random = RandomPTable(10, rows=10000)  # produces 10 columns named _1, _2, ...
max_ = Max(name="max_" + str(hash(random)))
max_.input[0] = random.output.result["_1", "_2", "_3"]  # hint ("_1", "_2", "_3")
pr = Print(proc=self.terse)
pr.input[0] = max_.output.result
random.scheduler().task_start()
```
Here, the hint "tells" to the `Max` module to compute the maximum only for columns "_1", "_2", "_3".
Otherwise, when no hint is provided, the maximum is computed for all the columns.

In addition to input and output slots, a module maintains a set of **parameters** that it uses internally.
Finally, some modules are **interactive** and can receive **events**, typically from the user interface or visualization interactions to **control** (stop, resume, step the execution) or **steer** the computation.

Most progressive programs are composed of existing modules, created with specific parameters and connected to form a specific program.
However, new modules can be programmed to implement a new function, to add an algorithm, a loader for a new file format, or a new visualization.
Programming a module is explained in the advanced section of this documentation.

## Running a Progressive Program

The easiest environment to run progressive programs is the JupyterLab notebook.
**ProgressiVis** comes with specified widgets, visualizations, and mechanisms to navigate a notebook in a non-linear way to follow the progression of modules.
Therefore, ProgressiVis offers two levels of programming, a low-level, as shown in the first example above, and a high-level designed for JupyterLab, more convenient, hiding boilerplate code and providing convenient widgets and navigation mechanisms inside JupyterLab to manage the non-sequential style of ProgressiVis programs.

Alternatively, progressive programs can be run in a _headless_ environment.
We also provide an experimental setup to run them behind a web server to create progressive applications without a notebook.
This setup is experimental and should be extended in the future.

## Communication between ProgressiVis and the Notebook

ProgressiVis runs using asynchronous functions. The communication between ProgressiVis and the notebook is done through callbacks and function calls.
Module callbacks are handy to update the environment outside of ProgressiVis.
For example, visualizing the heatmap shown in the first example works like this:
```python
import ipywidgets as ipw
from IPython.display import display

# Create an empty Image widget and display it in the notebook
img = ipw.Image(value=b'\x00', width=width, height=height)
display(img)

# Define a callback (not that it is `async`) run after the heatmap module is updated
async def _after_run(m: Module, run_number: int) -> None:
    assert isinstance(m, Heatmap)
    image = m.get_image_bin()  # get the image from the heatmap
    if image is not None:
        img.value = image  # Replace the displayed image with the new one

heatmap.on_after_run(_after_run)  # Install the callback
```

On the other direction, an external function can trigger changes in a ProgressiVis program in a few ways. There is a low-level mechanisms based on the method `Module.from_input(msg)` that allows communicating with modules. The module `Variable` is the simplest module designed to handle external events through `from_input`. It implementation of `from_input` expects a dictionary that is then propagated as data in its output slot in the progressive program. Most of the interactions proposed in ProgressiVis are done though `Variable` modules. Reusing the same declarations as in the examples above, we can add dynamic filtering to the data being progressively loaded with the following code:

```python
from progressivis import (
    CSVLoader, Histogram2D, ConstDict, Heatmap, PDict,
    BinningIndexND, RangeQuery2d, Variable
)
import progressivis.core.aio as aio

bnds_min = PDict({col_x: bounds.left, col_y: bounds.bottom})
bnds_max = PDict({col_x: bounds.right, col_y: bounds.top})
# Create a csv loader for the taxi data file
csv = CSVLoader(LARGE_TAXI_FILE, index_col=False, usecols=[col_x, col_y])
# Create an indexing module on the csv loader output columns
index = BinningIndexND()
# Creates one index per numeric column
index.input.table = csv.output.result[col_x, col_y]
# Create a querying module
query = RangeQuery2d(column_x=col_x, column_y=col_y)
# Variable modules allow to dynamically modify their values; here, the query ranges
var_min = Variable(name="var_min")
var_max = Variable(name="var_max")
query.input.lower = var_min.output.result
query.input.upper = var_max.output.result
query.input.index = index.output.result
query.input.min = index.output.min_out
query.input.max = index.output.max_out
# Create a module to compute the 2D histogram of the two columns specified
# with the given resolution
histogram2d = Histogram2D(col_x, col_y, xbins=RESOLUTION, ybins=RESOLUTION)
# Connect the module to the csv results and the min,max bounds to rescale
histogram2d.input.table = query.output.result
histogram2d.input.min = query.output.min
histogram2d.input.max = query.output.max
# Create a module to create an heatmap image from the histogram2d
heatmap = Heatmap()
# Connect it to the histogram2d
heatmap.input.array = histogram2d.output.result
heatmap.display_notebook()
csv.scheduler().task_start();
```

which produces the following dataflow graph:

```{eval-rst}
.. progressivis_dot:: ./userguide1.3.py
```

Controlling this progressive graph with jupyter widgets can be done like this:
```{eval-rst}
.. literalinclude:: ./userguide1.3-cont.py
   :linenos:
```

Line 6 and 7 use the `Module.from_input` method to initialize the
 value of `var_min` and `var_max` to `bnds_min` and `bnds_max` respectively.
 Then, two range sliders are created line 9 and line 21 to filter a range of values
 between the specified bounds of the visualization.
 The `observer()` function is attached as callback of these two sliders to collect the
 slider values and send them to `var_min` and `var_max` on line 37-39.
 Setting them in the callback will force the histogram to recompute with the new bounds and, in turn, trugger an update of the heatmap every time the sliders are moved.

Building a progressive visualization and making it interactive is conceptually easy with ProgressiVis, but can need long boilerplate code. To simplify the construction of complex loading, analysis, and visualization progressive pipelines, we provide a higher-level of abstraction in jupyter lab notebooks.  They are documented in the [visualization section](visualizations).

