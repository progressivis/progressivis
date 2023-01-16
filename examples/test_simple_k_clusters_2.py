"""
Clustering datasets may be found at
http://cs.joensuu.fi/sipu/datasets/
"""
from progressivis import Scheduler, Every  # , log_level
from progressivis.cluster import MBKMeans
from progressivis.io import CSVLoader
from progressivis.vis import MCScatterPlot
from progressivis.utils.psdict import PDict
from progressivis.core import aio
import numpy as np
from progressivis.datasets.random import (
    generate_random_multivariate_normal_csv as gen_csv,
)

try:
    s = scheduler
except NameError:
    s = Scheduler()
    # log_level(package="progressivis.cluster")

file_name = "/tmp/foobar.csv"
gen_csv(file_name, rows=999999, reset=True)  # , header='_0,_1', reset=False)
data = CSVLoader(
    file_name, skipinitialspace=True, header=None, index_col=False, scheduler=s
)
mbkmeans = MBKMeans(
    columns=["_0", "_1"],
    n_clusters=3,
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
# mbkmeans.input.table = data.output.table
mbkmeans.create_dependent_modules()
sp.move_point = mbkmeans.moved_center  # for input management


def myprint(d):
    if d["convergence"] != "unknown":
        print(d)
    else:
        print(".", end="")


prn = Every(scheduler=s, proc=print)
prn.input.df = mbkmeans.output.conv

if __name__ == "__main__":
    # data.start()
    # s.join()
    aio.run(s.start())
