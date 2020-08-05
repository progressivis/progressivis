"""
Clustering datasets may be found at
http://cs.joensuu.fi/sipu/datasets/
"""
from progressivis import Scheduler, Every#, log_level
from progressivis.cluster import MBKMeans, MBKMeansFilter
from progressivis.io import CSVLoader
from progressivis.vis import MCScatterPlot
from progressivis.datasets import get_dataset
from progressivis.stats import RandomTable
from progressivis.utils.psdict import PsDict
import pandas as pd
import numpy as np
import os.path
import tempfile
from progressivis.datasets.random import generate_random_multivariate_normal_csv as gen_csv
try:
    s = scheduler
except NameError:
    s = Scheduler()
    #log_level(package="progressivis.cluster")

#dir_name = tempfile.mkdtemp(prefix='progressivis_tmp_')
dir_name = os.path.join(tempfile.gettempdir(), 'progressivis_tmp_')
os.makedirs(dir_name, exist_ok=True)
file_name = os.path.join(dir_name, "foobar.csv")
gen_csv(file_name, rows=99999, reset=True) #, header='_0,_1', reset=False)
data = CSVLoader(file_name, skipinitialspace=True, header=None, index_col=False,scheduler=s)
n_clusters = 3
mbkmeans = MBKMeans(columns=['_0', '_1'], n_clusters=n_clusters, batch_size=100, tol=0.01, is_input=False, scheduler=s)
classes = []
for i in range(n_clusters):
    cname = f"k{i}"
    filt = MBKMeansFilter(i)
    filt.create_dependent_modules(mbkmeans, data, 'table')
    classes.append({'name': cname, 'x_column': '_0',
                    'y_column': '_1', 'sample': mbkmeans if i==0 else None,
                    'input_module': filt, 'input_slot': 'table'})

sp = MCScatterPlot(scheduler=s, classes=classes)
sp.create_dependent_modules()
for i in range(n_clusters):
    cname = f"k{i}"
    sp[cname].min_value._table = PsDict({'_0': -np.inf, '_1': -np.inf})
    sp[cname].max_value._table = PsDict({'_0': np.inf, '_1': np.inf})
mbkmeans.input.table = data.output.table
mbkmeans.create_dependent_modules()
sp.move_point = mbkmeans.moved_center # for input management

def myprint(d):
    if d['convergence']!='unknown':
        print(d)
    else:
        print('.', end='')

        
prn = Every(scheduler=s, proc=print)
prn.input.df = mbkmeans.output.conv

if __name__ == '__main__':
    #data.start()
    #s.join()
    aio.run(s.start())
