import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.core.combine_first import CombineFirst
from progressivis.datasets import get_dataset

import pandas as pd
import numpy as np
from pprint import pprint

def print_len(x):
    if x is not None:
        print len(x)


class TestCombineFirst(unittest.TestCase):
    def test_combine_first_dup(self):
        s=Scheduler()
        cst1=Constant(pd.DataFrame({'xmin': [1], 'xmax': [2]}), scheduler=s)
        cst2=Constant(pd.DataFrame({'ymin': [5], 'ymax': [6]}), scheduler=s)
        cst3=Constant(pd.DataFrame({'ymin': [3], 'ymax': [4]}), scheduler=s)
        cf=CombineFirst(scheduler=s)
        cf.input.df = cst1.output.df
        cf.input.df = cst2.output.df
        cf.input.df = cst3.output.df
        pr=Print(scheduler=s)
        pr.input.df = cf.output.df
        s.start()
        res = cf.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print res
        df = cf.df()
        last = df.loc[df.index[-1]]
        print last
        self.assertTrue(last['xmin']==1 and last['xmax']==2 and \
                        last['ymin']==5 and last['ymax']==6)

    def test_combine_first_nan(self):
        s=Scheduler()
        cst1=Constant(pd.DataFrame({'xmin': [1], 'xmax': [2]}), scheduler=s)
        cst2=Constant(pd.DataFrame({'ymin': [np.nan], 'ymax': [np.nan]}), scheduler=s)
        cst3=Constant(pd.DataFrame({'ymin': [3], 'ymax': [4]}), scheduler=s)
        cf=CombineFirst(scheduler=s)
        cf.input.df = cst1.output.df
        cf.input.df = cst2.output.df
        cf.input.df = cst3.output.df
        pr=Print(scheduler=s)
        pr.input.df = cf.output.df
        s.start()
        res = cf.trace_stats(max_runs=1)
        pd.set_option('display.expand_frame_repr', False)
        print res
        df = cf.df()
        last = df.loc[df.index[-1]]
        print last
        self.assertTrue(last['xmin']==1 and last['xmax']==2 and \
                        last['ymin']==3 and last['ymax']==4)
