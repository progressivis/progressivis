import unittest

from progressivis import Scheduler, Every, log_level
from progressivis.io import VECLoader
from progressivis.io import CSVLoader
from progressivis.metrics import PairwiseDistances
from progressivis.datasets import get_dataset

import numpy as np
from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances

import logging


def print_len(x):
    if x is not None:
        print x.shape

times = 0

def ten_times(scheduler, run_number):
    global times
    times += 1
    if times > 10:
        scheduler.stop()

class TestPairwiseDistances(unittest.TestCase):
    def NOsetUp(self):
        log_level(logging.DEBUG,'progressivis.metrics.pairwise')

    def NOtest_vec_distances(self):
        s=Scheduler()
        vec=VECLoader(get_dataset('warlogs'),scheduler=s)
        dis=PairwiseDistances(metric='cosine',scheduler=s)
        dis.input.df = vec.output.df
        dis.input.array = vec.output.array
        cnt = Every(proc=print_len,constant_time=True,scheduler=s)
        cnt.input.df = dis.output.dist
        global times
        times = 0
        s.start()
        df = vec.df()
        computed = dis.dist()
        self.assertEquals(computed.shape[0], len(df))
        truth = pairwise_distances(vec.toarray(), metric=dis._metric)
        self.assertTrue(np.allclose(truth, computed))

    def test_csv_distances(self):
        s=Scheduler()
        vec=CSVLoader(get_dataset('smallfile'),index_col=False,header=None,scheduler=s)
        dis=PairwiseDistances(metric='euclidean',scheduler=s)
        dis.input.df = vec.output.df
        cnt = Every(proc=print_len,constant_time=True,scheduler=s)
        cnt.input.df = dis.output.dist
        global times
        times = 0
        s.start(ten_times)
        df = vec.df()
        computed = dis.dist()
        #self.assertEquals(computed.shape[0], len(df))

        del df[CSVLoader.UPDATE_COLUMN]
        offset=0
        size=offset+5000
        truth = pairwise_distances(df.iloc[offset:size], metric=dis._metric)
        dist = computed[offset:size,offset:size]
        self.assertTrue(np.allclose(truth, dist,atol=1e-7)) # reduce tolerance

if __name__ == '__main__':
    unittest.main()
