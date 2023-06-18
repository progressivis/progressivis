# User Guide


*ProgressiVis* is a language and system implemented in Python that uses specific constructs to remain reactive and interactive all the time. If you are familiar with asynchronous programming or real time programming, you will be familiar with the need to follow strict disciplines to make sure a system is not blocked. ProgressiVis is also designed so that it never runs functions that take an unbounded amount of time, such as loading a large file from the network until completion.
In a traditional computation system, you don't worry much about the time taken when calling a function; loading a large file over the network is the price to pay for having it loaded and starting computations over its contents. In a progressive system, meant to be used interactively, when a function takes too long to complete, the user waits, get bored, and her attention drops.  ProgressiVis is designed to avoid this attention drop.

## Key concepts

ProgressiVis is a language and system implementing *progressive data analysis and visualization*.
Let's start with a simple non progressive program to introduce the concept.  Assume we want to find out what are the popular places to go in New York City. You can visualize all the NYC Yellow Taxi pickup places as a start. Using the Python Pandas library, you would write something like:
```python
import pandas as pd


TAXI_FILE = "https://raw.githubusercontent.com/bigdata-vandy/spark-taxi/master/yellow_tripdata_2015-01.csv"

df = pd.read_csv(TAXI_FILE, index_col=False)
df.plot.scatter(x="pickup_latitude", y="pickup_longitude")
```


In that case, you would only get the trips from January 2015. It would take you a few seconds to download the file and to visualize it. And you would see this:
![](images/userguide_1_bad.png)

which is probably not what you expected. The file is not very large but yet, you see a very common issue with real data: noise and outliers. In that particular case, one taxi driver has forgotten to stop his meter when driving to Florida, leading to a much too large bounding box for the data.

The most common way to deal with this outlier problem is to filter outlier points, typically the points outside the 2% quantiles up and down:
```python

plq = df.pickup_longitude.quantile([0.02, 0.98])

```
Now, you would see this:
![](images/userguide_1_ok.png)

This dataset used in this example is a sample of the real one; it is only 1000 lines long.  The real dataset contains 2 million lines (about 12Mb) for January only and 12 times more for the whole year 2015 (1.5 billion lines, 22Gb). It would take several minutes to download the data, and a few more seconds to visualize it.




## Synchronization of Modules

When multiple modules are computing values over the same table, they may become desynchronized, some may be lagging behind due different processing speed.
