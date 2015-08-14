import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.core.wait import Wait

import os
import csv
import numpy as np
import pandas as pd
from pprint import pprint

class TestStats(unittest.TestCase):
    filename='data/bigfile.csv'
    rows = 1000000
    cols = 30
    def setUp(self):
        log_level()
        self.scheduler = Scheduler()
        if os.path.exists(self.filename):
            return
        print "Generating %s for testing" % self.filename
        with open(self.filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for r in range(0,self.rows):
                row=list(np.random.rand(self.cols))
                writer.writerow(row)

    def test_stats(self):
        csv_module = CSVLoader(self.filename,id='test_read_csv',
                               index_col=False,header=None,chunksize=3000,
                               scheduler=self.scheduler)
        stats=Stats(1,id='test_stats', scheduler=self.scheduler)
        wait=Wait(id='wait', delay=3, scheduler=self.scheduler)
        wait.input.inp = csv_module.output.df
        #connect(csv_module, 'df', wait, 'inp')
        stats.input._params = wait.output.out
        #connect(wait, 'out', stats, '_params')
        import pdb
#        pdb.set_trace()
#        stats.describe()
        #connect(csv_module, 'df', stats, 'df')
        stats.input.df = csv_module.output.df
        pr = Print(id='print', scheduler=self.scheduler)
        #connect(stats, 'stats', pr, 'inp')
        pr.input.inp = stats.output.stats
        self.scheduler.start()
        s = stats.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print s

suite = unittest.TestLoader().loadTestsFromTestCase(TestStats)

if __name__ == '__main__':
    unittest.main()
