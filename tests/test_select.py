import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.core.select import Select
from progressivis.datasets import get_dataset

import pandas as pd
from pprint import pprint

def print_len(x):
    if x is not None:
        print len(x)

#log_level(package='progressivis.core.select')

class TestSelect(unittest.TestCase):
    def test_query_simple(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,force_valid_ids=True,scheduler=s)
        q=Select(scheduler=s)
        q.input.df = csv.output.df
        prlen = Every(proc=print_len, constant_time=True, scheduler=s)
        prlen.input.df = q.output.df
        s.start()
        self.assertEqual(len(q.df()), 1000000)

    def test_query(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,force_valid_ids=True,scheduler=s)
        cst=Constant(pd.DataFrame({'query': ['_1 < 0.5']}),scheduler=s)
        q=Select(scheduler=s)
        q.input.df = csv.output.df
        q.input.query = cst.output.df
        prlen = Every(proc=print_len, constant_time=True, scheduler=s)
        prlen.input.df = q.output.df
        s.start()
        self.assertTrue(len(q.df()) < 1000000)


if __name__ == '__main__':
    unittest.main()
