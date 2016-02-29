from progressivis import Scheduler, Every, Print
from progressivis.io import CSVLoader
from progressivis.stats import Stats, RandomTable

#log_level()

def filter(df):
    l = df['pickup_longitude']
    return df[(l < -70) & (l > -80) ]

try:
    s = scheduler
except:
    s = Scheduler()
#csv = CSVLoader('../nyc-taxi/yellow_tripdata_2015-10.csv.bz2', index_col=False,skipinitialspace=True,usecols=['pickup_longitude', 'pickup_latitude'], filter=filter, engine='c',scheduler=s)
csv = RandomTable(columns=['pickup_longitude', 'pickup_latitude'],rows=1000000, throttle=100000, scheduler=s)
long=Stats('pickup_longitude',scheduler=s)
long.input.df = csv.output.df
lat=Stats('pickup_latitude',scheduler=s)
lat.input.df = csv.output.df
pr = Print(scheduler=s)
pr.input.df = lat.output.stats
prlen = Every(scheduler=s)
prlen.input.df = csv.output.df

# if __name__=='__main__':
#     print "Starting"
#     csv.start()
