import unittest

from progressivis import Print, Scheduler
from progressivis.stats import IdxMax, Max, RandomTable

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
        max1=max.last_row(max.df(),remove_update=True)
        #print max1
        max2=idxmax.last_row(idxmax.max(),remove_update=True)
        #print max2
        self.assertTrue((max1==max2).all())

if __name__ == '__main__':
    unittest.main()
