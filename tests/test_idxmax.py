import unittest

from progressivis import Print, Scheduler
from progressivis.stats import IdxMax, IdxMin, Max, Min, RandomTable
from progressivis.core.utils import last_row

import pandas as pd
from pprint import pprint

def print_len(x):
    if x is not None:
        print len(x)


class TestIdxMax(unittest.TestCase):
    def test_idxmax(self):
        s=Scheduler()
        random = RandomTable(10, rows=10000,throttle=1000, scheduler=s)
        idxmax=IdxMax(scheduler=s)
        idxmax.input.df = random.output.df
        max=Max(scheduler=s)
        max.input.df = random.output.df
        pr=Print(scheduler=s)
        pr.input.df = idxmax.output.max
        s.start()
        max1=last_row(max.df(),remove_update=True)
        #print max1
        max2=last_row(idxmax.max(),remove_update=True)
        #print max2
        self.assertTrue((max1==max2).all())

    def test_idxmin(self):
        s=Scheduler()
        random = RandomTable(10, rows=10000,throttle=1000, scheduler=s)
        idxmin=IdxMin(scheduler=s)
        idxmin.input.df = random.output.df
        min=Min(scheduler=s)
        min.input.df = random.output.df
        pr=Print(scheduler=s)
        pr.input.df = idxmin.output.min
        s.start()
        min1=last_row(min.df(),remove_update=True)
        #print min1
        min2=last_row(idxmin.min(),remove_update=True)
        #print min2
        self.assertTrue((min1==min2).all())

if __name__ == '__main__':
    unittest.main()
