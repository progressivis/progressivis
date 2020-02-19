# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %gui asyncio
from progressivis.nbwidgets import *
sc = Scatterplot()
gr = ModuleGraph()

# +

import time
import pandas as pd
import copy
from progressivis.core import Scheduler, Every, JSONEncoderNp, asynchronize
from progressivis.table import Table
from progressivis.vis import MCScatterPlot
from progressivis.io import CSVLoader
#from progressivis.datasets import get_dataset
from progressivis.table.constant import Constant
import asyncio as aio
import functools as ft

def _quiet(x): pass


def _filter(df):
    pklon = df['pickup_longitude']
    pklat = df['pickup_latitude']
    dolon = df['dropoff_longitude']
    dolat = df['dropoff_latitude']
    return df[(pklon > -74.08) & (pklon < -73.5) & (pklat > 40.55) & (pklat < 41.00) &
                  (dolon > -74.08) & (dolon < -73.5) & (dolat > 40.55) & (dolat < 41.00)]

def _print_len(x):
    if x is not None:
        print(len(x))

def _cleanup(data):
    return JSONEncoderNp.cleanup(data)

def wait_for_change(widget, value):
    future = aio.Future()
    def getvalue(change):
        # make the new value available
        future.set_result(change.new)
        widget.unobserve(getvalue, value)
    widget.observe(getvalue, value)
    return future
           
 

def feed_widget(wg, val):
    wg.data = _cleanup(val)

    
class MyScatterPlot(MCScatterPlot):
    async def run(self, run_number):
        await super().run(run_number)
        aio.create_task(asynchronize(feed_widget, sc, self._json_cache))
        aio.create_task(asynchronize(feed_widget, gr, self.scheduler().to_json(short=False)))
        
                        

try:
    s = scheduler
except NameError:
    s = Scheduler()

#PREFIX= 'https://storage.googleapis.com/tlc-trip-data/2015/'
#SUFFIX= ''
PREFIX = '../../../nyc-taxi/'

SUFFIX = '.bz2'

URLS = [
    PREFIX+'yellow_tripdata_2015-01.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-02.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-03.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-04.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-05.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-06.csv'+SUFFIX,
]

FILENAMES = pd.DataFrame({'filename': URLS})
#import pdb;pdb.set_trace()
CST = Constant(Table('filenames', data=FILENAMES), scheduler=s)
CSV = CSVLoader(index_col=False, skipinitialspace=True,
                usecols=['pickup_longitude', 'pickup_latitude',
                             'dropoff_longitude', 'dropoff_latitude'],
                filter_=_filter, scheduler=s) # TODO: reimplement filter in read_csv.py

CSV.input.filenames = CST.output.table
PR = Every(scheduler=s, proc=_quiet)
PR.input.df = CSV.output.table
MULTICLASS = MyScatterPlot(scheduler=s, classes=[
    ('pickup', 'pickup_longitude', 'pickup_latitude'),
    ('dropoff', 'dropoff_longitude', 'dropoff_latitude')], approximate=True)
MULTICLASS.create_dependent_modules(CSV, 'table')

async def from_input():
    while True:
        await wait_for_change(sc, 'value')
        bounds = sc.value
        await MULTICLASS.min_value.from_input(bounds['min'])
        await MULTICLASS.max_value.from_input(bounds['max'])        

aio.create_task(s.start(coros=[from_input()]));
# -

import ipywidgets as ipw
tab = ipw.Tab()
tab.children = [sc, gr]
tab.set_title(0, 'Scatterplot')
tab.set_title(1, 'Module graph')
tab
