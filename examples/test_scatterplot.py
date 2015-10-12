from progressivis import *
from progressivis.vis import ScatterPlot
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset

def filter(df):
    l = df['pickup_longitude']
    return df[(l < -70) & (l > -80) ]

def print_len(x):
    if x is not None:
        print len(x)

#log_level()

MTScheduler.set_default()

s=MTScheduler()
#csv = CSVLoader('../nyc-taxi/yellow_tripdata_2014-10.csv', index_col=False,skipinitialspace=True,usecols=['pickup_longitude', 'pickup_latitude'], filter=filter, scheduler=s)
csv = CSVLoader(get_dataset('bigfile'),header=None,index_col=False,scheduler=s)
pr = Every(scheduler=s)
pr.input.inp = csv.output.df
scatterplot = ScatterPlot(1, 2, scheduler=s)
wait=scatterplot.create_scatterplot_modules()
wait.input.inp = csv.output.df

if __name__=='__main__':
    csv.start()
    s.thread.join()
    print len(csv.df())
