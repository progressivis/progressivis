"""
Test loading of nyc_taxis with dynamic queries.
"""
import pandas as pd
from progressivis import Scheduler, Every, Table, CSVLoader, Constant
from progressivis.vis import MCScatterPlot
import asyncio as aio


def _filter(df):
    lon = df['pickup_longitude']
    lat = df['pickup_latitude']
    return df[(lon > -74.08) & (lon < -73.5) & (lat > 40.55) & (lat < 41.00)]


def _print_len(x):
    if x is not None:
        print(len(x))


try:
    s = scheduler
except NameError:
    s = Scheduler()

#PREFIX= 'https://storage.googleapis.com/tlc-trip-data/2015/'
#SUFFIX= ''
PREFIX = '../nyc-taxi/'

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
CST = Constant(Table('filenames', data=FILENAMES), scheduler=s)
CSV = CSVLoader(skipinitialspace=True,
                usecols=['pickup_longitude', 'pickup_latitude'],
                filter_=_filter, scheduler=s)

CSV.input.filenames = CST.output.table
PR = Every(scheduler=s)
PR.input.df = CSV.output.table
SCATTERPLOT = MCScatterPlot(scheduler=s,
                                classes=[('Scatterplot',
                                              'pickup_longitude',
                                              'pickup_latitude')],
                                approximate=True)

SCATTERPLOT.create_dependent_modules(CSV, 'table')

async def coro(s):
    await aio.sleep(2)
    print("awake after 2 sec.")
    s.to_json()

if __name__ == '__main__':
    aio.run(s.start(coros=[coro(s), aio.sleep(3600)]))
    print(len(CSV.table()))
