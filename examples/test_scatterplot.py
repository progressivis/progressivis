from progressivis import Scheduler, Every
from progressivis.vis import ScatterPlot
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset

def filter_(df):
    l = df['pickup_longitude']
    return df[(l < -70) & (l > -80) ]

def print_len(x):
    if x is not None:
        print(len(x))

#log_level()

try:
    s = scheduler
except:
    s = Scheduler()

csv = CSVLoader(get_dataset('bigfile'),header=None,index_col=False,force_valid_ids=True,scheduler=s)
pr = Every(scheduler=s)
pr.input.df = csv.output.table
scatterplot = ScatterPlot(x_column='_1', y_column='_2', scheduler=s)
scatterplot.create_dependent_modules(csv,'table')

if __name__=='__main__':
    csv.start()
    s.join()
    print(len(csv.df()))
