from progressivis import Scheduler, Every, Print
from progressivis.io import CSVLoader
from progressivis.stats import Histogram2D, Min, Max
from progressivis.datasets import get_dataset

print "Loading test_histogram2d"
print "Type of default_scheduler is %s" % type(Scheduler.default)

csv = CSVLoader(get_dataset('bigfile'),index_col=False,header=None,engine='c')
pr = Every()
pr.input.df = csv.output.df
min = Min()
min.input.df = csv.output.df
max = Max()
max.input.df = csv.output.df
histogram2d=Histogram2D(1, 2, xbins=128, ybins=128)
histogram2d.input.df = csv.output.df
histogram2d.input.min = min.output.df
histogram2d.input.max = max.output.df
pr = Print(id='print')
pr.input.df = histogram2d.output.df

if __name__=='__main__':
    csv.start()
