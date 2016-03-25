import unittest

from progressivis import Print, Scheduler
from progressivis.stats import Var, RandomTable
from progressivis.core.utils import last_row

import pandas as pd
import numpy as np


class Testvar(unittest.TestCase):
    def test_var(self):
        s=Scheduler()
        random = RandomTable(1, rows=1000, scheduler=s)
        var=Var(scheduler=s)
        var.input.df = random.output.df
        pr=Print(scheduler=s)
        pr.input.df = var.output.df
        s.start()
        res1 = random.df()[1].var()
        res2 = last_row(var.df(), remove_update=True)
        #print 'res1:', res1
        #print 'res2:', res2
        self.assertTrue(np.allclose(res1, res2))

if __name__ == '__main__':
    unittest.main()
