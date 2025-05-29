from progressivis import Scheduler, Every, CSVLoader
from progressivis.vis import ScatterPlot
from progressivis.datasets import get_dataset


def filter_(df):
    lon = df['pickup_longitude']
    return df[(lon < -70) & (lon > -80)]


def print_len(x):
    if x is not None:
        print(len(x))

s = Scheduler()


csv = CSVLoader(get_dataset('bigfile'),header=None,force_valid_ids=True,scheduler=s)
pr = Every(scheduler=s)
pr.input.df = csv.output.table
scatterplot = ScatterPlot(x_column='_1', y_column='_2', scheduler=s)
scatterplot.create_dependent_modules(csv,'table')

if __name__=='__main__':
    csv.start()
    s.join()
    print(len(csv.df()))
