"""
Clustering datasets may be found at
http://cs.joensuu.fi/sipu/datasets/
"""
from progressivis import Scheduler, Every#, log_level
from progressivis.cluster import MBKMeans
from progressivis.io import CSVLoader
from progressivis.vis import MCScatterPlot
from progressivis.datasets import get_dataset
from progressivis.stats import RandomTable
import pandas as pd
import numpy as np
from progressivis.datasets.random import generate_random_multivariate_normal_csv as gen_csv
try:
    s = scheduler
except NameError:
    s = Scheduler()
    #log_level(package="progressivis.cluster")

file_name = "/tmp/foobar.csv"
gen_csv(file_name, rows=999999) #, header='_0,_1', reset=False)
data = CSVLoader(file_name, skipinitialspace=True, header=None, index_col=False,scheduler=s)
mbkmeans = MBKMeans(columns=['_0', '_1'], n_clusters=3, batch_size=100, tol=0.01, is_input=False, scheduler=s)
sp = MCScatterPlot(scheduler=s, classes=[('Scatterplot', '_0', '_1', mbkmeans)], approximate=True)
sp.create_dependent_modules(data,'table')

mbkmeans.input.table = data.output.table

prn = Every(scheduler=s, proc=print)
prn.input.df = mbkmeans.output.conv

if __name__ == '__main__':
    #data.start()
    #s.join()
    aio.run(s.start())
