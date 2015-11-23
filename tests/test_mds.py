import unittest

from progressivis import *
from progressivis.io import VECLoader
from progressivis.io import CSVLoader
from progressivis.metrics import PairwiseDistances
from progressivis.manifold import MDS
from progressivis.datasets import get_dataset

import logging

def print_len(x):
    if x is not None:
        print len(x)

def ten_times(scheduler, run_number):
    if run_number > 10:
        scheduler.stop()

class TestMDS(unittest.TestCase):
    def setUp(self):
        log_level(logging.INFO,'progressivis')

    # def test_MDS_vec(self):
    #     vec=VECLoader(get_dataset('warlogs'))
    #     dis=PairwiseDistances(metric='cosine')
    #     dis.input.df = vec.output.df
    #     dis.input.array = vec.output.array
    #     cnt = Every(proc=print_len,constant_time=True)
    #     cnt.input.df = dis.output.df
    #     vec.start()

    def test_MDS_csv(self):
        s=Scheduler()
        vec=CSVLoader(get_dataset('smallfile'),index_col=False,header=None,scheduler=s)
        dis=PairwiseDistances(metric='euclidean',scheduler=s)
        dis.input.df = vec.output.df
        cnt = Every(proc=print_len,constant_time=True,scheduler=s)
        cnt.input.df = dis.output.distance
        s.start(ten_times)

if __name__ == '__main__':
    unittest.main()
