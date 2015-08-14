import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.core.merge import Merge
from progressivis.datasets import get_dataset

import pandas as pd
from pprint import pprint

class TestMerge(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler()

    def test_merge(self):
        csv_module = CSVLoader(get_dataset('bigfile'),
                               id='test_read_csv',
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
                Print(id='print', scheduler=self.scheduler), 'inp')
        self.scheduler.start()
        s = merge.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print s

if __name__ == '__main__':
    unittest.main()
