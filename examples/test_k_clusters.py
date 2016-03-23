from progressivis import Scheduler, Every, log_level
from progressivis.cluster import MBKMeans
from progressivis.io import CSVLoader
from progressivis.stats import Min, Max, Histogram2D
from progressivis.vis import Heatmap, ScatterPlot
from progressivis.datasets import get_dataset

"""
Clustering datasets may be found at
https://cs.joensuu.fi/sipu/datasets/
"""

try:
    s = scheduler
except:
    s = Scheduler()
    #log_level(package="progressivis.cluster")

data = CSVLoader(get_dataset('cluster:s3'),sep='    ',skipinitialspace=True,header=None,index_col=False,scheduler=s)
mbkmeans = MBKMeans(columns=[0, 1], n_clusters=15, batch_size=100, scheduler=s)
mbkmeans.input.df = data.output.df
prn = Every(scheduler=s)
prn.input.df = mbkmeans.output.df
sp = ScatterPlot(0,1, scheduler=s)
sp.move_point = mbkmeans # for input management
#sp.create_dependent_modules(mbkmeans,'centroids')
# Create modules by hand rather than with the utility.
# We show the cluster centroids on the scatterplot and the
# data as a heatmap

# histogram2d
histogram2d = Histogram2D(0, 1, scheduler=s)
histogram2d.input.df = data.output.df
min = Min([0,1], scheduler=s)
max = Max([0,1], scheduler=s)
min.input.df = data.output.df
max.input.df = data.output.df
histogram2d.input.min = min.output.df
histogram2d.input.max = max.output.df
# heatmap
heatmap = Heatmap(filename='heatmap%d.png', history=100, scheduler=s)
heatmap.input.array = histogram2d.output.df
# scatterplot
sp.input.heatmap = heatmap.output.heatmap
sp.input.df = mbkmeans.output.df

if __name__ == '__main__':
    data.start()
    s.join()
