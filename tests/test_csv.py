import unittest

from progressivis import *
from progressivis.io import CSVLoader

import os
import csv
import numpy as np
from pprint import pprint

log_level()
    
class TestProgressiveLoadCSV(unittest.TestCase):
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

    def tearDownNO(self):
        try:
            os.remove(self.filename)
        except:
            pass

    def test_read_csv(self):
        module=CSVLoader(self.filename,id='test_read_csv',scheduler=self.scheduler,index_col=False,header=None)
        self.assertTrue(module.df() is None)
        module.run(0)
        df = module.df()
        self.assertFalse(df is None)
        l = len(df)
        self.assertEqual(l, len(df[module.update_timestamps()==module.last_update()]))
        cnt = 1
        
        while not module.is_terminated():
            module.run(cnt)
            cnt += 1
            s = module.trace_stats(max_runs=1)
            df = module.df()
            ln = len(df)
            print "Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), ln)
            self.assertEqual(ln-l, len(df[module.update_timestamps()==module.last_update()]))
            l =  ln
        s = module.trace_stats(max_runs=1)
        print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        self.assertEqual(len(module.df()), self.rows)
        df2 = module.df().groupby([Module.UPDATE_COLUMN])
        self.assertEqual(cnt, len(df2))

suite = unittest.TestLoader().loadTestsFromTestCase(TestProgressiveLoadCSV)

if __name__ == '__main__':
    unittest.main()
