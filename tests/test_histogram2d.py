import unittest

from progressivis import *
from progressivis.io import CSVLoader
from progressivis.stats import Histogram2D
from progressivis.vis import Heatmap
from progressivis.datasets import get_dataset

import os
import glob
import csv
import numpy as np
import pandas as pd
from pprint import pprint

class TestHistogram2D(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler()

    def tearDown(self):
        StorageManager.default.end()

    def test_histogram2d(self):
        csv = CSVLoader(get_dataset('bigfile'),
                        id='csv',
                        index_col=False,header=None,
                        scheduler=self.scheduler)
        histogram2d=Histogram2D(1, 2, # columns are called 1..30
                                id='histogram2d',
                                xbins=100,
                                ybins=100,
                                scheduler=self.scheduler)
        histogram2d.input.df = csv.output.df
        heatmap=Heatmap(id='heatmap', filename='histo_%03d.png',
                       scheduler=self.scheduler)
        heatmap.input.array = histogram2d.output.histogram2d
        pr = Print(id='print', scheduler=self.scheduler)
        #pr.input.inp = heatmap.output.heatmap
        pr.input.inp = histogram2d.output.histogram2d
        self.scheduler.start()
        s = histogram2d.trace_stats(max_runs=1)
        #print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        pd.set_option('display.expand_frame_repr', False)
        print s

if __name__ == '__main__':
    unittest.main()
