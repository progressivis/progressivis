import unittest

from progressivis import *
from progressivis.io import VECLoader
from progressivis.io import CSVLoader
from progressivis.metrics import PairwiseDistances
from progressivis.datasets import get_dataset

import logging

def print_len(x):
    if x is not None:
        print len(x)

class TestPairwiseDistances(unittest.TestCase):
    def setUp(self):
        log_level(logging.INFO,'progressivis')

# def test_vec_distances(self):
    #     vec=VECLoader(get_dataset('warlogs'))
    #     dis=PairwiseDistances(metric='cosine')
    #     dis.input.df = vec.output.df
    #     dis.input.array = vec.output.array
    #     cnt = Every(proc=print_len,constant_time=True)
    #     cnt.input.inp = dis.output.df
    #     vec.start()

    def test_csv_distances(self):
        scheduler=Scheduler()
        vec=CSVLoader(get_dataset('smallfile'),index_col=False,header=None,scheduler=scheduler)
        dis=PairwiseDistances(metric='euclidean',scheduler=scheduler)
        dis.input.df = vec.output.df
        cnt = Every(proc=print_len,constant_time=True,scheduler=scheduler)
        cnt.input.inp = dis.output.df
        scheduler.start()

if __name__ == '__main__':
    unittest.main()
