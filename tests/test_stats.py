import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.core.wait import Wait
from progressivis.datasets import get_dataset

import pandas as pd

class TestStats(unittest.TestCase):
    def setUp(self):
        log_level()
        self.scheduler = Scheduler()

    def test_stats(self):
        csv_module = CSVLoader(get_dataset('bigfile'),
                               id='test_read_csv',
                               index_col=False,header=None,
                               scheduler=self.scheduler)
        stats=Stats(1,id='test_stats', scheduler=self.scheduler)
        wait=Wait(id='wait', delay=3, scheduler=self.scheduler)
        wait.input.inp = csv_module.output.df
        #connect(csv_module, 'df', wait, 'inp')
        stats.input._params = wait.output.out
        #connect(wait, 'out', stats, '_params')
        import pdb
        #connect(csv_module, 'df', stats, 'df')
        stats.input.df = csv_module.output.df
        pr = Print(id='print', scheduler=self.scheduler)
        #connect(stats, 'stats', pr, 'inp')
        pr.input.inp = stats.output.stats
        self.scheduler.start()
        s = stats.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print s

if __name__ == '__main__':
    unittest.main()
