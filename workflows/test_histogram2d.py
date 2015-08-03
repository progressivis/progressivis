from progressive import *
from progressive.io import CSVLoader
from progressive.stats import Histogram2d
from progressive.vis import Heatmap

print "Loading test_histogram2d"
print "Type of default_scheduler is %s" % type(Scheduler.default)

csv = CSVLoader('data/bigfile.csv',id='csv',index_col=False,header=None,chunksize=3000)
histogram2d=Histogram2d(1, 2, id='histogram2d', xbins=100, ybins=100)
histogram2d.input.df = csv.output.df
heatmap=Heatmap(id='heatmap', filename='histo_%03d.png')
heatmap.input.array = histogram2d.output.histogram2d
pr = Print(id='print')
pr.input.inp = histogram2d.output.histogram2d
