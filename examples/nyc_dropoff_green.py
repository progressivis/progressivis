from progressivis import MTScheduler, Every, Constant
from progressivis.stats import Histogram2D, Min, Max
from progressivis.io import CSVLoader
from progressivis.vis import Heatmap

import pandas as pd

RESOLUTION=1024

def filter(df):
    lon = df['Dropoff_longitude']
    lat = df['Dropoff_latitude']
    return df[(lon>-74.10)&(lon<-73.7)&(lat>40.60)&(lat<41)]

def print_len(x):
    if x is not None:
        print len(x)

#log_level() #package='progressivis.stats.histogram2d')

try:
    s = scheduler
except:
    s = MTScheduler()

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
cst = Constant(df=filenames, scheduler=s)
csv = CSVLoader(index_col=False,skipinitialspace=True,usecols=['Dropoff_longitude', 'Dropoff_latitude'], filter=filter, scheduler=s)
csv.input.filenames = cst.output.df
min = Min(scheduler=s)
min.input.df = csv.output.df
max = Max(scheduler=s)
max.input.df = csv.output.df
histogram2d = Histogram2D('Dropoff_longitude', 'Dropoff_latitude', xbins=RESOLUTION, ybins=RESOLUTION, scheduler=s)
histogram2d.input.df = csv.output.df
histogram2d.input.min = min.output.df
histogram2d.input.max = max.output.df
heatmap = Heatmap(filename='nyc_Dropoff_green%d.png', history=5, scheduler=s)
heatmap.input.array = histogram2d.output.df

if __name__=='__main__':
    s.start()
