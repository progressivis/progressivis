import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.core.join import Join
from progressivis.datasets import get_dataset

import pandas as pd
from pprint import pprint

def print_len(x):
    if x is not None:
        print len(x)


class TestJoin(unittest.TestCase):
    def test_join(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,scheduler=s)
        stat1=Stats(1, scheduler=s)
        stat1.input.df = csv.output.df
        stat2=Stats(2, scheduler=s)
        stat2.input.df = csv.output.df
        join=Join(scheduler=s)
        join.input.df = stat1.output.stats
        join.input.df = stat2.output.stats
        pr=Print(scheduler=s)
        pr.input.inp = join.output.df
        prlen = Every(proc=print_len, constant_time=True, scheduler=s)
        prlen.input.inp = csv.output.df
        s.start()
        res = join.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print res

    def test_join_simple(self):
        s=Scheduler()
        cst1=Constant(pd.DataFrame({'xmin': [1], 'xmax': [2]}), scheduler=s)
        cst2=Constant(pd.DataFrame({'ymin': [3], 'ymax': [4]}), scheduler=s)
        join=Join(scheduler=s)
        join.input.df = cst1.output.df
        join.input.df = cst2.output.df
        pr=Print(scheduler=s)
        pr.input.inp = join.output.df
        s.start()
        res = join.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print res
        df = join.df()
        last = df.loc[df.index[-1]]
        self.assertTrue(last['xmin']==1 and last['xmax']==2 and \
                        last['ymin']==3 and last['ymax']==4)


if __name__ == '__main__':
    unittest.main()
