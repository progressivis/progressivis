import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.core.wait import Wait
from progressivis.datasets import get_dataset

import pandas as pd

class TestStats(unittest.TestCase):
    def test_stats(self):
        s=Scheduler()
        csv_module = CSVLoader(get_dataset('smallfile'), index_col=False,header=None,
                               scheduler=s)
        stats=Stats(1,id='test_stats', scheduler=s)
        wait=Wait(id='wait', delay=3, scheduler=s)
        wait.input.df = csv_module.output.df
        #connect(csv_module, 'df', wait, 'df')
        stats.input._params = wait.output.df
        #connect(wait, 'df', stats, '_params')
        #connect(csv_module, 'df', stats, 'df')
        stats.input.df = csv_module.output.df
        pr = Print(id='print', scheduler=s)
        #connect(stats, 'stats', pr, 'inp')
        pr.input.df = stats.output.stats
        s.start()
        s = stats.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print s

if __name__ == '__main__':
    unittest.main()
