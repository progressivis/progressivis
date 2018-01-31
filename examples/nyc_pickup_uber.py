from progressivis import Scheduler, Every
from progressivis.table.constant import Constant
from progressivis.stats import Histogram2D, Min, Max
from progressivis.io import CSVLoader
from progressivis.vis import Heatmap
from progressivis.table import Table

import pandas as pd

RESOLUTION=1024

def filter_(df):
    lon = df['Lon']
    lat = df['Lat']
    return df[(lon>-74.10)&(lon<-73.7)&(lat>40.60)&(lat<41)]

def print_len(x):
    if x is not None:
        print(len(x))

#log_level() #package='progressivis.stats.histogram2d')

try:
    s = scheduler
except:
    s = Scheduler()

PREFIX= '../nyc-taxi/'
SUFFIX= '.bz2'

URLS = [
    PREFIX+'uber-raw-data-apr14.csv'+SUFFIX,
    PREFIX+'uber-raw-data-may14.csv'+SUFFIX,
    PREFIX+'uber-raw-data-jun14.csv'+SUFFIX,
    PREFIX+'uber-raw-data-jul14.csv'+SUFFIX,
    PREFIX+'uber-raw-data-sep14.csv'+SUFFIX,
]

filenames = pd.DataFrame({'filename': URLS})
cst = Constant(table=Table('filenames', data=filenames), scheduler=s)
csv = CSVLoader(index_col=False,skipinitialspace=True,usecols=['Lon', 'Lat'], filter_=filter_, scheduler=s)
csv.input.filenames = cst.output.table
min = Min(scheduler=s)
min.input.table = csv.output.table
max = Max(scheduler=s)
max.input.table = csv.output.table
histogram2d = Histogram2D('Lon', 'Lat', xbins=RESOLUTION, ybins=RESOLUTION, scheduler=s)
histogram2d.input.table = csv.output.table
histogram2d.input.min = min.output.table
histogram2d.input.max = max.output.table
heatmap = Heatmap(filename='nyc_pickup_uber%d.png', history=5, scheduler=s)
heatmap.input.array = histogram2d.output.table

if __name__=='__main__':
    s.start()
