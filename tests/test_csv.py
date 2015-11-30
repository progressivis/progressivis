import unittest

from progressivis import Constant, Scheduler
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset

import pandas as pd

import logging, sys

class TestProgressiveLoadCSV(unittest.TestCase):
    def setUp(self):
        self.logger=logging.getLogger('progressivis.core')
        self.saved=self.logger.getEffectiveLevel()
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler(stream=sys.stdout)
        self.logger.addHandler(ch)

    def tearDown(self):
        self.logger.setLevel(self.saved)

    def runit(self, module):
        module.run(1)
        df = module.df()
        self.assertFalse(df is None)
        l = len(df)
        self.assertEqual(l, len(df[df[module.UPDATE_COLUMN]==module.last_update()]))
        cnt = 2
        
        while not module.is_zombie():
            module.run(cnt)
            cnt += 1
            s = module.trace_stats(max_runs=1)
            df = module.df()
            ln = len(df)
            #print "Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), ln)
            self.assertEqual(ln-l, len(df[df[module.UPDATE_COLUMN]==module.last_update()]))
            l =  ln
        s = module.trace_stats(max_runs=1)
        print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        return cnt

    def test_read_csv(self):
        s=Scheduler()
        module=CSVLoader(get_dataset('bigfile'), index_col=False, header=None, scheduler=s)
        self.assertTrue(module.df() is None)
        s.start()
        self.assertEqual(len(module.df()), 1000000)

    def test_read_multiple_csv(self):
        s=Scheduler()
        filenames = pd.DataFrame({'filename': [get_dataset('smallfile'), get_dataset('smallfile')]})
        cst = Constant(df=filenames, scheduler=s)
        csv = CSVLoader(index_col=False, header=None, scheduler=s)
        csv.input.filenames = cst.output.df
        csv.start()
        self.assertEqual(len(csv.df()), 60000)


if __name__ == '__main__':
    unittest.main()
