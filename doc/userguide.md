# User Guide


**ProgressiVis** is a language and system implementing **progressive data analysis and visualization**.
If you are familiar with asynchronous programming or real time programming, you will be familiar with the need to follow strict disciplines to make sure a system is not blocked. ProgressiVis is also designed so that it never runs functions that take an unbounded amount of time, such as loading a large file from the network until completion.
In a traditional computation system, you don't worry much about the time taken when calling a function; loading a large file over the network is the price to pay for having it loaded and starting computations over its contents. In a progressive system, meant to be used interactively, when a function takes too long to complete, the user waits, get bored, and her attention drops.  ProgressiVis is designed to avoid this attention drop.

## Key concepts

**ProgressiVis** uses specific constructs to remain reactive and interactive all the time.
Let's start with a simple non progressive program to introduce the concept.  Assume we want to find out what are the popular places to go in New York City. You can visualize all the NYC Yellow Taxi pickup places as a start. Using the Python Pandas library, you would write something like:
```python
import pandas as pd


TAXI_FILE = "https://raw.githubusercontent.com/bigdata-vandy/spark-taxi/master/yellow_tripdata_2015-01.csv"

df = pd.read_csv(TAXI_FILE, index_col=False)
df.plot.scatter(x="pickup_latitude", y="pickup_longitude")
```


In that case, you would only get the trips from January 2015. It would take you a few seconds to download the file and to visualize it. And you would see this:
![](images/userguide_1_bad.png)

which is probably not what you expected. You see a very common issue with real data: noise and outliers. In that particular case, one taxi driver has forgotten to stop his meter when driving to Florida, leading to a much too large bounding box for the data.

The most common way to deal with this outlier problem is to filter outlier points, typically the points outside the 2% quantiles up and down:
```python

plq = df.pickup_longitude.quantile([0.02, 0.98])

```
Now, you would see this:
![](images/userguide_1_ok.png)

This dataset used in this example is a sample of the real one; it is only 1000 lines long.  The real dataset contains 2 million lines (about 12Mb) for January only and 12 times more for the whole year 2015 (1.5 billion lines, 22Gb). It would take several minutes to download the data, and a few more seconds to visualize it.

## Progressive Visualization

With ProgressiVis, you could also load the data and fix it, but you don't need to wait for the file to be loaded to visualize it, you can do it on the go:

```python
import progressivis as pv

TAXI_FILE = "https://raw.githubusercontent.com/bigdata-vandy/spark-taxi/master/yellow_tripdata_2015-01.csv"

csv = pv.CSVLoader(TAXI_FILE, index_col=False)
scatterplot = pv.ScatterPlot(x_column='dropoff_latitude',
                             y_column='dropoff_longitude')
scatterplot.create_dependent_modules(csv,'table')
csv.start()
```

## Main Components

In ProgressiVis, a program is run by a `Scheduler`. Only one instance of `Scheduler` exists (except tests). A progressive program is internally represented as a dataflow of progressive modules (simply called **modules** in this documentation). The dataflow is a directed network with no cycle (a directed acyclic graph or DAG).

A `Module` represents the equivalent of a function in a progressive
program.  It is made of input and output slots; one output slot of a
module can be connected to several input slots of other modules. Some
input slots are **optional**, and others are **mandatory**. Furthermore, a slot
is **typed** since it carries data between the modules.  A module with no input slot is a **source module**, and a module with no output slot is a **sink module**.

In addition to input and output slots, a module maintains a set of **parameters** that it uses internally. Finally, some modules are **interactive** and can receive **events**, typically from the user interface or visualization interactions to **control** (stop, resume, step the execution) or **steer** the computation.

Most progressive program is composed of existing modules, created with specific parameters and connected to form a specific program. However, new modules can be programmed to implement a new function to add an algorithm, a loader for a new file format, or a new visualization. Programming a module is explained in the advanced section of this documentation.

## Running a Progressive Program

The easiest environment to run progressive programs is the JupyterLab notebook. ProgressiVis comes with specified widgets, visualizations, and mechanisms to navigate a notebook in a non-linear way to follow the progression of modules.

Alternatively, progressive programs can be run in a _headless_ environment. We also provide an experimental setup to run them behind a web server to create progressive applications without a notebook. This setup is experimental and should be extended in the future.


