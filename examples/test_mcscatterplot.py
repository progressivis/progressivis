"""
Test loading of nyc_taxis with dynamic queries.
"""
import time
import six
import pandas as pd
import copy
from progressivis.core import Scheduler, Every
from progressivis.table import Table
from progressivis.vis import MCScatterPlot
from progressivis.io import CSVLoader
#from progressivis.datasets import get_dataset
from progressivis.table.constant import Constant
import asyncio as aio


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

#log_level() #package='progressivis.stats.histogram2d')

try:
    s = scheduler
except NameError:
    s = Scheduler()

#PREFIX= 'https://storage.googleapis.com/tlc-trip-data/2015/'
#SUFFIX= ''
PREFIX = '../nyc-taxi/'
if six.PY3:
    SUFFIX = '.bz2'
else:
    SUFFIX = '.gz'

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
PR = Every(scheduler=s)
PR.input.df = CSV.output.table
MULTICLASS = MCScatterPlot(scheduler=s, classes=[
    ('pickup', 'pickup_longitude', 'pickup_latitude'),
    ('dropoff', 'dropoff_longitude', 'dropoff_latitude')], approximate=True)
MULTICLASS.create_dependent_modules(CSV, 'table')

async def coro(s):
    await aio.sleep(2)
    print("awake after 2 sec.")
    s.to_json()


if __name__ == '__main__':
    aio.run(s.start(coros=[coro(s), aio.sleep(3600)]))
    print(len(CSV.table()))
