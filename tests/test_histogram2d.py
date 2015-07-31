import unittest

from progressive import *
from progressive.io import CSVLoader
from progressive.stats import Histogram2d
from progressive.vis import Heatmap

import os
import csv
import numpy as np
import pandas as pd
from pprint import pprint

class TestHistogram2d(unittest.TestCase):
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

    def test_histogram2d(self):
        csv_module = CSVLoader(self.filename,id='csv',
                               index_col=False,header=None,chunksize=3000,
                               scheduler=self.scheduler)
        module=Histogram2d(1, 2, # columns are called 1..30
                           id='histogram2d',
                           scheduler=self.scheduler)
        connect(csv_module, 'df', module, 'df')
        connect(module, 'histogram2d',
                Print(id='print', scheduler=self.scheduler), 'inp')
        heatmap=Heatmap(id='heatmap', filename='histo.png',
                       scheduler=self.scheduler)
        connect(module, 'histogram2d', heatmap, 'array')
        self.scheduler.run()
        s = module.trace_stats(max_runs=1)
        #print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        pd.set_option('display.expand_frame_repr', False)
        print s

suite = unittest.TestLoader().loadTestsFromTestCase(TestHistogram2d)

if __name__ == '__main__':
    unittest.main()
