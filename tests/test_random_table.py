import unittest

from progressivis import Every
from progressivis import Scheduler
from progressivis.stats import RandomTable

import pandas as pd

def print_len(x):
    if x is not None:
        print len(x)


class TestRandomTable(unittest.TestCase):
    def test_randome_table(self):
        s=Scheduler()
        module=RandomTable(['a', 'b'], rows=10000, scheduler=s)
        self.assertEqual(module.df().columns[0],'a')
        self.assertEqual(module.df().columns[1],'b')
        self.assertEqual(len(module.df().columns), 3) # add the UPDATE_COLUMN
        prlen = Every(proc=print_len, constant_time=True, scheduler=s)
        prlen.input.inp = module.output.df
        s.start()
        self.assertEqual(len(module.df()), 10000)
        self.assertFalse(module.df()['a'].hasnans())
        self.assertFalse(module.df()['b'].hasnans())

    def test_randome_table2(self):
        s=Scheduler()
         # produces more than 4M rows per second on my laptop
        module=RandomTable(10, rows=10000000, force_valid_ids=True, scheduler=s)
        self.assertEqual(len(module.df().columns), 11) # add the UPDATE_COLUMN
        self.assertEqual(module.df().columns[0],'_1')
        self.assertEqual(module.df().columns[1],'_2')
        prlen = Every(proc=print_len, constant_time=True, scheduler=s)
        prlen.input.inp = module.output.df
        s.start()
        self.assertEqual(len(module.df()), 10000000)
        self.assertFalse(module.df()['_1'].hasnans())
        self.assertFalse(module.df()['_2'].hasnans())
