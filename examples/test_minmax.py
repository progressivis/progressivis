from progressivis import Scheduler, Every, Print, RandomTable
from progressivis.stats.stats import Stats

#log_level()

def filter_(df):
    long = df['pickup_longitude']
    return df[(long < -70) & (long > -80)]

s = Scheduler()
##########################################################################################
#INPUT MODULE -> Random initialization of a tab with two columns
#csv = CSVLoader('../nyc-taxi/yellow_tripdata_2015-10.csv.bz2', index_col=False,skipinitialspace=True,usecols=['pickup_longitude', 'pickup_latitude'], filter_=filter_, engine='c',scheduler=s)
csv = RandomTable(columns=['pickup_longitude', 'pickup_latitude'],rows=1000000, throttle=100000, scheduler=s)

##########################################################################################
#COMPUTATIONAL MODULE -> Computes statistics on a given column (min & max)
# 'pickup_longitude' is a parameter that is interpreted by the module, in this case the column that is analyzed
long_=Stats('pickup_longitude',scheduler=s)
#input and output are slots of the the Stats and RandomTable modules respectively
long_.input.table = csv.output.table

##########################################################################################
#COMPUTATIONAL MODULE -> Computes statistics on a given column (min & max)
lat=Stats('pickup_latitude',scheduler=s)
lat.input.table = csv.output.table

##########################################################################################
# Printer takes as input the output of 'lat' which is the Stat module acting on the 'pickup_latitude' column
# It prints its input on the CONSOLE
pr = Print(scheduler=s)
pr.input.df = lat.output.stats

##########################################################################################
# Every takes as input the output of the random table
# It is a dummy class
prlen = Every(scheduler=s)
prlen.input.df = csv.output.table

# if __name__=='__main__':
#     print "Starting"
#     csv.start()
