import unittest

from progressivis import *
from progressivis.io import CSVLoader
from progressivis.stats import Histogram2D, Min, Max
from progressivis.vis import Heatmap
from progressivis.datasets import get_dataset

import os
import glob
import csv
import numpy as np
import pandas as pd
from pprint import pprint

class TestHistogram2D(unittest.TestCase):
#    def setUp(self):
#        self.scheduler = MTScheduler()
#        log_level()
#        self.scheduler = Scheduler()

    def tearDown(self):
        StorageManager.default.end()

    def test_histogram2d(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,scheduler=s)
        min = Min(scheduler=s)
        min.input.df = csv.output.df
        max = Max(scheduler=s)
        max.input.df = csv.output.df
        histogram2d=Histogram2D(1, 2, xbins=100, ybins=100,scheduler=s) # columns are called 1..30
        histogram2d.input.df = csv.output.df
        histogram2d.input.min = min.output.df
        histogram2d.input.max = max.output.df
        heatmap=Heatmap(filename='histo_%03d.png',scheduler=s)
        heatmap.input.array = histogram2d.output.df
        #pr = Print(scheduler=s)
        pr = Every(scheduler=s)
        #pr.input.inp = heatmap.output.heatmap
        #pr.input.inp = histogram2d.output.df
        pr.input.inp = csv.output.df
        csv.scheduler().start()
        #self.scheduler.thread.join()
        s = histogram2d.trace_stats()
        #print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        pd.set_option('display.expand_frame_repr', False)
        print s

if __name__ == '__main__':
    #import cProfile
    #cProfile.run("unittest.main()", 'prof')
    unittest.main()
