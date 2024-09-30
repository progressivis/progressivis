from progressivis import (
    Scheduler,
    Constant,
    Histogram2D,
    CSVLoader,
    Heatmap,
    Table
)

import pandas as pd

RESOLUTION = 1024

bounds_min = {'pickup_latitude': 40.60, 'pickup_longitude': -74.10}
bounds_max = {'pickup_latitude': 41.00, 'pickup_longitude': -73.70}


def filter_(df):
    lon = df['pickup_longitude']
    lat = df['pickup_latitude']
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
    PREFIX+'yellow_tripdata_2015-01.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-02.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-03.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-04.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-05.csv'+SUFFIX,
    PREFIX+'yellow_tripdata_2015-06.csv'+SUFFIX,
]

filenames = pd.DataFrame({'filename': URLS})
cst = Constant(Table('filenames', data=filenames), scheduler=s)
csv = CSVLoader(index_col=False,skipinitialspace=True,usecols=['pickup_longitude', 'pickup_latitude'], filter_=filter_, scheduler=s)
csv.input.filenames = cst.output.table
#min = Min(scheduler=s)
#min.input.df = csv.output.df
#max = Max(scheduler=s)
#max.input.df = csv.output.df
min = Constant(table=Table('bounds_min', data=pd.DataFrame([bounds_min])), scheduler=s)
max = Constant(table=Table('bounds_min', data=pd.DataFrame([bounds_max])), scheduler=s)
histogram2d = Histogram2D('pickup_longitude', 'pickup_latitude', xbins=RESOLUTION, ybins=RESOLUTION, scheduler=s)
histogram2d.input.table = csv.output.table
histogram2d.input.min = min.output.table
histogram2d.input.max = max.output.table
heatmap = Heatmap(filename='nyc_pickup_yellow%d.png', history=5, scheduler=s)
heatmap.input.array = histogram2d.output.table

if __name__=='__main__':
    s.start()
