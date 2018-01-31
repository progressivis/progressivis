"""
Clustering datasets may be found at
https://cs.joensuu.fi/sipu/datasets/
"""
from progressivis import Scheduler, Every#, log_level
from progressivis.cluster import MBKMeans
from progressivis.io import CSVLoader
from progressivis.vis import ScatterPlot
from progressivis.datasets import get_dataset

try:
    s = scheduler
except NameError:
    s = Scheduler()
    #log_level(package="progressivis.cluster")

data = CSVLoader(get_dataset('cluster:s1'),sep='\\s+',skipinitialspace=True,header=None,index_col=False,scheduler=s)
mbkmeans = MBKMeans(columns=['_0', '_1'], n_clusters=15, batch_size=100, is_input=False, scheduler=s)
mbkmeans.input.table = data.output.table
prn = Every(scheduler=s)
prn.input.df = mbkmeans.output.table
sp = ScatterPlot('_0','_1', scheduler=s)

sp.move_point = mbkmeans # for input management
sp.create_dependent_modules(data,'table', sample=None, select=mbkmeans)

if __name__ == '__main__':
    data.start()
    s.join()
