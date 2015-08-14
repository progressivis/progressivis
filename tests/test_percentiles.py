import unittest

from progressive import *
from progressivis.stats import Percentiles
from progressivis.io import CSVLoader

import os
import csv
import numpy as np
import pandas as pd
from pprint import pprint

class TestPercentiles(unittest.TestCase):
    rows = 30000
    cols = 10
    filename='data/smallfile.csv'
    def setUp(self):
        self.scheduler = Scheduler()
        if not os.path.exists(self.filename):
            print "Generating %s for testing" % self.filename
            with open(self.filename, 'w') as csvfile:
                writer = csv.writer(csvfile)
                for r in range(0,self.rows):
                    row=list(np.random.rand(self.cols))
                    writer.writerow(row)
        self.csv_module = CSVLoader(self.filename,id='test_read_csv',
                                    index_col=False,header=None,chunksize=3000,
                                    scheduler=self.scheduler)

    def test_percentile(self):
        module=Percentiles(1,id='test_percentile',
                           percentiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                           scheduler=self.scheduler)
        module.describe()
        self.csv_module.describe()
        connect(self.csv_module, 'df', module, 'df')
        connect(module, 'percentiles',
                Print(id='print', scheduler=self.scheduler), 'inp')
        self.scheduler.start()
        s = module.trace_stats(max_runs=1)
        #print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        pd.set_option('display.expand_frame_repr', False)
        print s

suite = unittest.TestLoader().loadTestsFromTestCase(TestPercentiles)

if __name__ == '__main__':
    unittest.main()
