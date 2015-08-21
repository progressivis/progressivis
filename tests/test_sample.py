import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.stats import Sample
from progressivis.datasets import get_dataset

import pandas as pd
from pprint import pprint
import logging



class TestSample(unittest.TestCase):
#    def setUp(self):
#        log_level(logging.INFO)

    def test_sample(self):
        csv    = CSVLoader(get_dataset('bigfile'),index_col=False,header=None)
        smp = Sample(n=10)
        smp.input.df = csv.output.df
        prt = Print()
        prt.input.inp = smp.output.sample
        csv.scheduler().start()


if __name__ == '__main__':
    unittest.main()
