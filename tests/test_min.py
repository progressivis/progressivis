import unittest

from progressivis import Print, Scheduler
from progressivis.stats import Min, Max, RandomTable

import pandas as pd
from pprint import pprint

def print_len(x):
    if x is not None:
        print len(x)


class TestMin(unittest.TestCase):
    def test_min(self):
        s=Scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        min=Min(scheduler=s)
        min.input.df = random.output.df
        pr=Print(scheduler=s)
        pr.input.inp = min.output.df
        s.start()

    def test_max(self):
        s=Scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        max=Max(scheduler=s)
        max.input.df = random.output.df
        pr=Print(scheduler=s)
        pr.input.inp = max.output.df
        s.start()

if __name__ == '__main__':
    unittest.main()
