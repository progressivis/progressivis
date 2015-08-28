import unittest

from progressivis import *
from progressivis.io import VECLoader
from progressivis.metrics import PairwiseDistances
from progressivis.datasets import get_dataset

def print_len(x):
    if x is not None:
        print len(x)

class TestPairwiseDistances(unittest.TestCase):
    def test_pairwise_distances(self):
        vec=VECLoader(get_dataset('warlogs'))
        dis=PairwiseDistances(metric='cosine')
        dis.input.df = vec.output.df
        dis.input.array = vec.output.array
        cnt = Every(proc=print_len,constant_time=True)
        cnt.input.inp = dis.output.df
        vec.start()

if __name__ == '__main__':
    unittest.main()
