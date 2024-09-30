"""
Clustering datasets may be found at
http://cs.joensuu.fi/sipu/datasets/
"""
from progressivis import Scheduler, Every, CSVLoader
from progressivis.cluster import MBKMeans
from progressivis.vis import MCScatterPlot
from progressivis.datasets import get_dataset
from progressivis.core import aio

try:
    s = scheduler
except NameError:
    s = Scheduler()
    #log_level(package="progressivis.cluster")

data = CSVLoader(get_dataset('cluster:s1'),sep='\\s+',skipinitialspace=True,header=None,index_col=False,scheduler=s)
mbkmeans = MBKMeans(columns=['_0', '_1'], n_clusters=15, batch_size=100, is_input=False, scheduler=s)
sp = MCScatterPlot(scheduler=s, classes=[('Scatterplot', '_0', '_1', mbkmeans)])
sp.create_dependent_modules(data,'table')

mbkmeans.input.table = data.output.table

prn = Every(scheduler=s)
prn.input.df = mbkmeans.output.table

if __name__ == '__main__':
    #data.start()
    #s.join()
    aio.run(s.start())
