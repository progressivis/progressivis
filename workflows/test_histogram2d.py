from progressive import *
from progressive.io import CSVLoader
from progressive.stats import Histogram2D
from progressive.vis import Heatmap

print "Loading test_histogram2d"
print "Type of default_scheduler is %s" % type(Scheduler.default)

csv = CSVLoader('data/bigfile.csv',id='csv',low_memory=False,index_col=False,header=None)
histogram2d=Histogram2D(1, 2, id='histogram2d', xbins=100, ybins=100)
histogram2d.input.df = csv.output.df
pr = Print(id='print')
pr.input.inp = histogram2d.output.histogram2d
