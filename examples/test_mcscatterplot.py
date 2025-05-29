"""
Test loading of nyc_taxis with dynamic queries.
"""
import pandas as pd
from progressivis import Scheduler, Every, Table, CSVLoader, Constant
from progressivis.vis import MCScatterPlot
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

SUFFIX = '.bz2'
"""
URLS = [
    PREFIX+'yellow_tripdata_2015-01.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-02.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-03.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-04.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-05.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-06.csv'+SUFFIX,
]
"""
URLS = [f"https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2015-0{n}.csv" for n in range(1,7)]
FILENAMES = pd.DataFrame({'filename': URLS})
#import pdb;pdb.set_trace()
CST = Constant(Table('filenames', data=FILENAMES), scheduler=s)
CSV = CSVLoader(skipinitialspace=True,
                usecols=['pickup_longitude', 'pickup_latitude',
                             'dropoff_longitude', 'dropoff_latitude'],
                filter_=_filter, scheduler=s) # TODO: reimplement filter in read_csv.py

CSV.input.filenames = CST.output.result
PR = Every(scheduler=s)
PR.input.df = CSV.output.result
MULTICLASS = MCScatterPlot(scheduler=s, classes=[
    ('pickup', 'pickup_longitude', 'pickup_latitude'),
    ('dropoff', 'dropoff_longitude', 'dropoff_latitude')], approximate=True)
MULTICLASS.create_dependent_modules(CSV, 'result')

async def coro(s):
    await aio.sleep(2)
    print("awake after 2 sec.")
    s.to_json()

if __name__ == '__main__':
    aio.run(s.start(coros=[coro(s), aio.sleep(3600)]))
    print(len(CSV.table()))
