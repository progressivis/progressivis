"""
Test loading of nyc_taxis with dynamic queries.
"""
import time
import six
import pandas as pd

from progressivis.core import Scheduler, Every
from progressivis.table import Table
from progressivis.vis import ScatterPlot
from progressivis.io import CSVLoader
#from progressivis.datasets import get_dataset
from progressivis.table.constant import Constant

def _filter(df):
    lon = df['pickup_longitude']
    lat = df['pickup_latitude']
    return df[(lon > -74.08) & (lon < -73.5) & (lat > 40.55) & (lat < 41.00)]

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
CST = Constant(Table('filenames', data=FILENAMES), scheduler=s)
CSV = CSVLoader(index_col=False, skipinitialspace=True,
                usecols=['pickup_longitude', 'pickup_latitude'],
                filter_=_filter, scheduler=s)

CSV.input.filenames = CST.output.table
PR = Every(scheduler=s)
PR.input.df = CSV.output.table
SCATTERPLOT = ScatterPlot('pickup_longitude', 'pickup_latitude', scheduler=s, approximate=True)
SCATTERPLOT.create_dependent_modules(CSV, 'table')
#s.set_interaction_opts(starving_mods=[SCATTERPLOT.sample, SCATTERPLOT.histogram2d], max_iter=3, max_time=1.5)
#s.set_interaction_opts(max_time=1.5)
s.set_interaction_opts(max_iter=3)
if __name__ == '__main__':
    s.start()
    while True:
        time.sleep(2)
        s.to_json()
        SCATTERPLOT.to_json() # simulate a web query
        SCATTERPLOT.get_image()
    s.join()
    print(len(CSV.table()))
