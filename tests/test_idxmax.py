import unittest

from progressivis import Print, Scheduler
from progressivis.stats import IdxMax, RandomTable

import pandas as pd
from pprint import pprint

def print_len(x):
    if x is not None:
        print len(x)


class TestIdxMax(unittest.TestCase):
    def test_idxmax(self):
        s=Scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        max=IdxMax(scheduler=s)
        max.input.df = random.output.df
        pr=Print(scheduler=s)
        pr.input.df = max.output.max
        s.start()

if __name__ == '__main__':
    unittest.main()
