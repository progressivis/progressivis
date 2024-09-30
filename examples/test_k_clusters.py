"""
Clustering datasets may be found at
https://cs.joensuu.fi/sipu/datasets/
"""
from progressivis import (
    Scheduler, Every,
    CSVLoader,
    PDict
)
from progressivis.vis import MCScatterPlot
from progressivis.datasets import get_dataset
from progressivis.core import aio
from progressivis.cluster import MBKMeans
import numpy as np

try:
    s = scheduler
except NameError:
    s = Scheduler()
    # log_level(package="progressivis.cluster")

data = CSVLoader(
    get_dataset("cluster:s1"),
    sep="\\s+",
    skipinitialspace=True,
    header=None,
    index_col=False,
    scheduler=s,
)
mbkmeans = MBKMeans(
    columns=["_0", "_1"],
    n_clusters=15,
    batch_size=100,
    tol=0.01,
    is_input=False,
    scheduler=s,
)
sp = MCScatterPlot(scheduler=s, classes=[("Scatterplot", "_0", "_1", mbkmeans)])
sp.create_dependent_modules(data, "table")
sp["Scatterplot"].min_value._table = PDict({"_0": -np.inf, "_1": -np.inf})
sp["Scatterplot"].max_value._table = PDict({"_0": np.inf, "_1": np.inf})
mbkmeans.input.table = sp["Scatterplot"].range_query_2d.output.table
mbkmeans.create_dependent_modules()
prn = Every(scheduler=s, proc=print)
prn.input.df = mbkmeans.output.conv

sp.move_point = mbkmeans.moved_center  # for input management


if __name__ == "__main__":
    # data.start()
    # s.join()
    aio.run(s.start())
