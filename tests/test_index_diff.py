import unittest

import pandas as pd
import numpy as np

from progressivis.core.index_diff import *

def random_range():
    start = np.random.randint(100)
    end = start + np.random.randint(200)
    return pd.RangeIndex(start, end)

class TestIndexDiff(unittest.TestCase):
    def test_index_difference(self):
        i1 = pd.RangeIndex(0,100)
        i2 = pd.RangeIndex(10,110)

        self.assertTrue(index_difference(i1,i2).equals(pd.RangeIndex(0,10)))
        self.assertTrue(index_difference(i2,i1).equals(pd.RangeIndex(100,110)))

    def test_index_difference2(self):
        for i in range(1000):
            i1 = random_range()
            i2 = random_range()
            self.assertTrue((index_difference(i1,i2)==i1.difference(i2)).all())
