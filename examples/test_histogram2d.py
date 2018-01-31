from progressivis import Scheduler, Every, Print
from progressivis.io import CSVLoader
from progressivis.stats import Histogram2D, Min, Max
from progressivis.datasets import get_dataset
from progressivis.vis import Heatmap

print("Loading test_histogram2d")
print("Type of default_scheduler is %s" % type(Scheduler.default))

csv = CSVLoader(get_dataset('bigfile'),index_col=False,header=None,engine='c')
pr = Every()
pr.input.df = csv.output.table
min_ = Min()
min_.input.table = csv.output.table
max_ = Max()
max_.input.table = csv.output.table
histogram2d=Histogram2D('_1', '_2', xbins=128, ybins=128)
histogram2d.input.table = csv.output.table
histogram2d.input.min = min_.output.table
histogram2d.input.max = max_.output.table
# heatmap
heatmap=Heatmap(filename='histo_%03d.png')
heatmap.input.array = histogram2d.output.table
pr = Print(id='print')
pr.input.df = csv.output.table


if __name__=='__main__':
    csv.start()
