import unittest

from progressivis import *
from progressivis.io import CSVLoader
from progressivis.stats import Histogram1D, Min, Max
from progressivis.datasets import get_dataset

import os
import glob
import csv
import numpy as np
import pandas as pd
from pprint import pprint

import logging
logging.basicConfig(level=logging.WARNING)

class TestHistogram1D(unittest.TestCase):

    def tearDown(self):
        StorageManager.default.end()

    def test_histogram1d(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,scheduler=s)
        min = Min(scheduler=s)
        min.input.df = csv.output.df
        max = Max(scheduler=s)
        max.input.df = csv.output.df
        histogram1d=Histogram1D(2, scheduler=s) # columns are called 1..30
        histogram1d.input.df = csv.output.df
        histogram1d.input.min = min.output.df
        histogram1d.input.max = max.output.df
   
        #pr = Print(scheduler=s)
        pr = Every(scheduler=s)
        pr.input.df = histogram1d.output.df
        csv.scheduler().start()
        #self.scheduler.thread.join()
        s = histogram1d.trace_stats()
        #print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        pd.set_option('display.expand_frame_repr', False)
        print s

if __name__ == '__main__':
    unittest.main()
