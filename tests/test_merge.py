import unittest

from progressive import *
from progressive.stats import Stats
from progressive.io import CSVLoader
from progressive.core.merge import Merge

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
        module1=Stats(1,id='test_stats_1', scheduler=self.scheduler)
        connect(csv_module, 'df', module1, 'df')
        module2=Stats(2,id='test_stats_2', scheduler=self.scheduler)
        connect(csv_module, 'df', module2, 'df')
        merge=Merge(id='merge', scheduler=self.scheduler)
        connect(module1, 'stats', merge, 'df')
        connect(module2, 'stats', merge, 'df')
        connect(merge, 'df',
                Print(id='print', scheduler=self.scheduler), 'in')
        self.scheduler.run()
        s = merge.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print s


if __name__ == '__main__':
    unittest.main()
