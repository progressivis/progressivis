import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.vis import ScatterPlot
from progressivis.datasets import get_dataset

import pandas as pd
from pprint import pprint
import logging

def print_len(x):
    if x is not None:
        print len(x)

def idle_proc(s, x):
    s.stop()

class TestScatterPlot(unittest.TestCase):
#    def setUp(self):
#        log_level(logging.INFO,'progressivis')

    def test_scatterplot(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('bigfile'),index_col=False,header=None,scheduler=s)
        sp = ScatterPlot(x_column=1, y_column=2, scheduler=s)
        wait = sp.create_scatterplot_modules()
        wait.input.df = csv.output.df
        cnt = Every(proc=print_len,constant_time=True,scheduler=s)
        cnt.input.inp = csv.output.df
        prt = Print(scheduler=s)
        prt.input.inp = sp.histogram2d.output.df
        csv.scheduler().start(None,idle_proc)
        self.assertEquals(len(csv.df()), 1000000)


if __name__ == '__main__':
    unittest.main()
