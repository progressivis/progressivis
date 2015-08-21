import unittest

from progressivis import *
from progressivis.stats import Percentiles
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset

import pandas as pd

class TestPercentiles(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler()

    def test_percentile(self):
        csv_module = CSVLoader(get_dataset('smallfile'),
                               id='test_read_csv',
                               index_col=False,header=None,
                               scheduler=self.scheduler)
        module=Percentiles(1,id='test_percentile',
                           percentiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                           scheduler=self.scheduler)
        module.describe()
        csv_module.describe()
        connect(csv_module, 'df', module, 'df')
        connect(module, 'percentiles',
                Print(id='print', scheduler=self.scheduler), 'inp')
        self.scheduler.start()
        s = module.trace_stats(max_runs=1)
        #print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        pd.set_option('display.expand_frame_repr', False)
        print s

if __name__ == '__main__':
    unittest.main()
