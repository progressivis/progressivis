import unittest

from progressive import *
from progressive.stats import Stats
from progressive.io import CSVLoader
from progressive.core.wait import Wait

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
        module=Stats(1,id='test_stats', scheduler=self.scheduler)
        wait=Wait(id='wait', delay=3, scheduler=self.scheduler)
        connect(csv_module, 'df', wait, 'inp')
        connect(wait, 'out', module, '_params')
        module.describe()
        connect(csv_module, 'df', module, 'df')
        connect(module, 'stats',
                Print(id='print', scheduler=self.scheduler), 'inp')
        self.scheduler.run()
        s = module.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print s

suite = unittest.TestLoader().loadTestsFromTestCase(TestStats)

if __name__ == '__main__':
    unittest.main()
