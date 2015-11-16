import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.core.filter import Filter
from progressivis.datasets import get_dataset

import pandas as pd
from pprint import pprint

def print_len(x):
    if x is not None:
        print len(x)

#log_level(package='progressivis.core.filter')

class TestFilter(unittest.TestCase):
    def test_filter_simple(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,force_valid_ids=True,scheduler=s)
        f=Filter(scheduler=s)
        f.input.df = csv.output.df
        prlen = Every(proc=print_len, constant_time=True, scheduler=s)
        prlen.input.inp = f.output.df
        s.start()
        self.assertEqual(len(f.df()), 1000000)

    def test_filter(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,force_valid_ids=True,scheduler=s)
        cst=Constant(pd.DataFrame({'filter': ['_1 < 0.5']}),scheduler=s)
        f=Filter(scheduler=s)
        f.input.df = csv.output.df
        f.input.filter = cst.output.df
        prlen = Every(proc=print_len, constant_time=True, scheduler=s)
        prlen.input.inp = f.output.df
        s.start()
        self.assertTrue(len(f.df()) < 1000000)


if __name__ == '__main__':
    unittest.main()
