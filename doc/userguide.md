# User Guide


**ProgressiVis** is a language and system implementing **progressive data analysis and visualization**.
`ProgressiVis` is designed so that it never blocks in functions during unbounded amount of time, such as loading a large file from the network until completion.
In a traditional computation system, you don't worry much about the time taken when calling a function; loading a large file over the network is the price to pay for having it loaded and starting computations over its contents. In a progressive system, meant to be used interactively, when a function takes too long to complete, the user waits, get bored, and her attention drops.  `ProgressiVis` is designed to avoid this attention drop.
If you are familiar with asynchronous programming or real time programming, you will be familiar with the need to follow strict disciplines to make sure a system is not blocking.  This discipline is implemented everywhere in **progressiVis** with a specific "progressive" semantics.

## Key concepts

**ProgressiVis** uses specific constructs to remain reactive and interactive all the time.
Let's start with a simple progressive program to introduce the concept.  Assume we want to find out what are the popular places to go in New York City.
We can download the New York Taxi dataset that contain all the taxi trips in 2015 and 2016, including the pickup and drop-off positions, looking for hot-spots.

With `ProgressiVis`, we don't need to wait for the file to be fully loaded to visualize it, we can do it on the go, as with this simple low-level ProgressiVis program:

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

## Main Components

In `ProgressiVis`, a program is run by a `Scheduler`. Only one instance of `Scheduler` exists (except in tests).
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
`ProgressiVis` comes with specified widgets, visualizations, and mechanisms to navigate a notebook in a non-linear way to follow the progression of modules.
Therefore, ProgressiVis offers two levels of programming, a low-level, as shown in the first example above, and a high-level designed for JupyterLab, more convenient, hiding boilerplate code and providing convenient widgets and navigation mechanisms inside JupyterLab to manage the non-sequential style of ProgressiVis programs.

Alternatively, progressive programs can be run in a _headless_ environment.
We also provide an experimental setup to run them behind a web server to create progressive applications without a notebook.
This setup is experimental and should be extended in the future.


