from progressivis import Scheduler, Every
from progressivis.cluster import MBKMeans
from progressivis.io import CSVLoader
from progressivis.vis import ScatterPlot

"""
Clustering datasets may be found at
https://cs.joensuu.fi/sipu/datasets/
"""

try:
    s = scheduler
except:
    s = Scheduler()

data = CSVLoader('s3.txt',sep=',',header=None,index_col=False,scheduler=s)
mbkmeans = MBKMeans(columns=[0, 1], n_clusters=15, batch_size=100)
mbkmeans.input.df = data.output.df
prn = Every(scheduler=s)
prn.input.df = mbkmeans.output.df
sp = ScatterPlot(0,1)
sp.create_dependent_modules(mbkmeans,'df')

if __name__ == '__main__':
    data.start()
    s.thread.join()
