import unittest

from progressivis import Print, Scheduler
from progressivis.stats import Min, Max, RandomTable
from progressivis.core.utils import last_row

import pandas as pd
import numpy as np

class TestMinMax(unittest.TestCase):
    def test_min(self):
        s=Scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        min=Min(scheduler=s)
        min.input.df = random.output.df
        pr=Print(scheduler=s)
        pr.input.df = min.output.df
        s.start()
        res1 = random.df()[random.columns.difference([random.UPDATE_COLUMN])].min()
        res2 = last_row(min._df, remove_update=True)
        self.assertTrue(np.allclose(res1, res2))

    def test_max(self):
        s=Scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        max=Max(scheduler=s)
        max.input.df = random.output.df
        pr=Print(scheduler=s)
        pr.input.df = max.output.df
        s.start()
        res1 = random.df()[random.columns.difference([random.UPDATE_COLUMN])].max()
        res2 = last_row(max.df(), remove_update=True)
        self.assertTrue(np.allclose(res1, res2))

if __name__ == '__main__':
    unittest.main()
