from . import ProgressiveTest

from progressivis import Print
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.core.wait import Wait
from progressivis.datasets import get_dataset

import numpy as np

class TestStats(ProgressiveTest):
#    def setUp(self):
#        log_level(logging.DEBUG,'progressivis.core')

    def test_stats(self):
        s = self.scheduler()
        csv_module = CSVLoader(get_dataset('smallfile'), index_col=False,header=None,
                               scheduler=s)
        stats=Stats('_1', name='test_stats', scheduler=s)
        wait=Wait(name='wait', delay=3, scheduler=s)
        wait.input.inp = csv_module.output.table
        stats.input._params = wait.output.out
        stats.input.table = csv_module.output.table
        pr = Print(proc=self.terse, name='print', scheduler=s)
        pr.input.df = stats.output.stats
        s.start()
        s.join()
        table = csv_module.table()
        stable = stats.table()
        last = stable.last()
        tmin = table['_1'].min()
        self.assertTrue(np.isclose(tmin, last['__1_min']))
        tmax = table['_1'].max()
        self.assertTrue(np.isclose(tmax, last['__1_max']))

if __name__ == '__main__':
    ProgressiveTest.main()
