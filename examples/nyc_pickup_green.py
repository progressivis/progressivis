from progressivis import (
    Scheduler,
    Constant,
    Histogram2D,
    Min, Max,
    CSVLoader,
    Heatmap,
    Table
)

import pandas as pd

RESOLUTION = 1024


def filter_(df):
    lon = df['Pickup_longitude']
    lat = df['Pickup_latitude']
    return df[(lon>-74.10)&(lon<-73.7)&(lat>40.60)&(lat<41)]


def print_len(x):
    if x is not None:
        print(len(x))

#log_level() #package='progressivis.stats.histogram2d')

s = Scheduler()

#PREFIX= 'https://storage.googleapis.com/tlc-trip-data/2015/'
#SUFFIX= ''
PREFIX= '../nyc-taxi/'
SUFFIX= '.bz2'

URLS = [
    PREFIX+'green_tripdata_2015-01.csv'+SUFFIX,
    PREFIX+'green_tripdata_2015-02.csv'+SUFFIX,
    PREFIX+'green_tripdata_2015-03.csv'+SUFFIX,
    PREFIX+'green_tripdata_2015-04.csv'+SUFFIX,
    PREFIX+'green_tripdata_2015-05.csv'+SUFFIX,
    PREFIX+'green_tripdata_2015-06.csv'+SUFFIX,
]

filenames = pd.DataFrame({'filename': URLS})
cst = Constant(Table('filenames', data=filenames), scheduler=s)
csv = CSVLoader(skipinitialspace=True,usecols=['Pickup_longitude', 'Pickup_latitude'], filter_=filter_, scheduler=s)
csv.input.filenames = cst.output.table
min = Min(scheduler=s)
min.input.table = csv.output.table
max = Max(scheduler=s)
max.input.table = csv.output.table
histogram2d = Histogram2D('Pickup_longitude', 'Pickup_latitude', xbins=RESOLUTION, ybins=RESOLUTION, scheduler=s)
histogram2d.input.table = csv.output.table
histogram2d.input.min = min.output.table
histogram2d.input.max = max.output.table
heatmap = Heatmap(filename='nyc_pickup_green%d.png', history=5, scheduler=s)
heatmap.input.array = histogram2d.output.table

if __name__=='__main__':
    s.start()
