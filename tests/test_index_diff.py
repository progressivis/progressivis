from . import ProgressiveTest

import pandas as pd
import numpy as np

from progressivis.core.index_diff import IndexDifference, NIL_INDEX

def random_range():
    start = np.random.randint(100)
    end = start + np.random.randint(200)
    return pd.RangeIndex(start, end)

class TestIndexDiff(ProgressiveTest):
    def _do_test_diff(self, i1,i2,expect1=None,expect2=None):
        d=IndexDifference(i1,i2)
        if expect1 is None:
            expect1=i1.difference(i2)
        if not d.equals(expect1):
            print(i1, i2)
            self.assertTrue(False)
        d=IndexDifference(i2,i1)
        if expect2 is None:
            expect2=i2.difference(i1)
        if not d.equals(expect2):
            print(i2, i1)
            self.assertTrue(False)
    
    def test_index_difference(self):
        self._do_test_diff(pd.RangeIndex(62,106), pd.RangeIndex(62,106), NIL_INDEX, NIL_INDEX)
        self._do_test_diff(pd.RangeIndex(62,106), pd.RangeIndex(65,89))
        self._do_test_diff(pd.RangeIndex(84,238), pd.RangeIndex(3,52))
        self._do_test_diff(pd.RangeIndex(0,100), pd.RangeIndex(10,110))

    def test_index_difference2(self):
        for i in range(1000):
            i1 = random_range()
            i2 = random_range()
            self._do_test_diff(i1,i2)

