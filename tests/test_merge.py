import unittest

from progressivis import Every, Scheduler, Constant, Print
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.core.merge import Merge
from progressivis.datasets import get_dataset

import pandas as pd
from pprint import pprint

def print_len(x):
    if x is not None:
        print len(x)


class TestMerge(unittest.TestCase):
    def test_merge(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,scheduler=s)
        stat1=Stats(1, scheduler=s)
        stat1.input.df = csv.output.df
        stat2=Stats(2, scheduler=s)
        stat2.input.df = csv.output.df
        merge=Merge(left_index=True,right_index=True,scheduler=s)
        merge.input.df = stat1.output.stats
        merge.input.df = stat2.output.stats
        pr=Print(scheduler=s)
        pr.input.df = merge.output.df
        prlen = Every(proc=print_len, constant_time=True, scheduler=s)
        prlen.input.df = csv.output.df
        s.start()
        res = merge.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print res

    def test_merge_simple(self):
        s=Scheduler()
        cst1=Constant(pd.DataFrame({'xmin': [1], 'xmax': [2]}), scheduler=s)
        cst2=Constant(pd.DataFrame({'ymin': [3], 'ymax': [4]}), scheduler=s)
        merge=Merge(left_index=True,right_index=True,scheduler=s)
        merge.input.df = cst1.output.df
        merge.input.df = cst2.output.df
        pr=Print(scheduler=s)
        pr.input.df = merge.output.df
        s.start()
        res = merge.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print res
        df = merge.df()
        last = df.loc[df.index[-1]]
        self.assertTrue(last['xmin']==1 and last['xmax']==2 and \
                        last['ymin']==3 and last['ymax']==4)

if __name__ == '__main__':
    unittest.main()
