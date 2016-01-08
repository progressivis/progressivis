import unittest
from time import sleep

from progressivis import *
from progressivis.io import CSVLoader
from progressivis.stats import Sample
from progressivis.datasets import get_dataset

import pandas as pd
import numpy as np


class TestScheduler(unittest.TestCase):

    def test_scheduler(self):
        s = Scheduler()
        csv = CSVLoader(get_dataset('bigfile'),index_col=False,header=None,scheduler=s)

        smp = Sample(n=10,scheduler=s)
        smp.input.df = csv.output.df
        prt = Print(scheduler=s)
        prt.input.df = smp.output.df

        csv.scheduler().start()

        sleep(1)
        self.assertTrue(csv.scheduler().is_running())

        smp2 = Sample(n=15, scheduler=s)
        smp2.input.df = csv.output.df


if __name__ == '__main__':
    unittest.main()
