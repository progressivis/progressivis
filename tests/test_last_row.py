import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset

import pandas as pd
from pprint import pprint

def print_len(x):
    if x is not None:
        print len(x)


class TestLastRow(unittest.TestCase):
    def test_join(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,scheduler=s)
        stat1=Stats(1, scheduler=s)
        stat1.input.df = csv.output.df
        stat2=Stats(2, scheduler=s)
        stat2.input.df = csv.output.df
        lr1 = LastRow(scheduler=s)
        lr1.input.df = stat1.output.stats
        lr2 = LastRow(scheduler=s)
        lr2.input.df = stat2.output.stats
        join=Join(scheduler=s)
        join.input.df = lr1.output.df
        join.input.df = lr2.output.df
        pr=Print(scheduler=s)
        pr.input.df = join.output.df
        prlen = Every(proc=print_len, constant_time=True, scheduler=s)
        prlen.input.df = csv.output.df
        s.start()
        res = join.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        last = join.df()
        df = csv.df()
        self.assertTrue(last.at[0,'1.min']==df[1].min() and last.at[0,'1.max']==df[1].max() and \
                        last.at[0,'2.min']==df[2].min() and last.at[0,'2.max']==df[2].max())

        print res

    def test_join_simple(self):
        s=Scheduler()
        cst1=Constant(pd.DataFrame({'xmin': [1], 'xmax': [2]}), scheduler=s)
        cst2=Constant(pd.DataFrame({'ymin': [3], 'ymax': [4]}), scheduler=s)
        join=Join(scheduler=s)
        join.input.df = cst1.output.df
        join.input.df = cst2.output.df
        pr=Print(scheduler=s)
        pr.input.df = join.output.df
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
