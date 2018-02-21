from . import ProgressiveTest

from progressivis.storage.range import RangeDataset, RangeError

import numpy as np

class TestRange(ProgressiveTest):
    def test_range(self):
        with self.assertRaises(ValueError):
            r = RangeDataset('range', shape=(10,10))
        with self.assertRaises(TypeError):
            r = RangeDataset('range', shape=(10,), dtype=np.float32)
        
        r = RangeDataset('range', shape=(10,))
        self.assertEqual(r.shape, (10,))
        self.assertEqual(r.size, 10)
        self.assertEqual(r.dtype, np.int)
        self.assertEqual(r[9], 9)
        self.assertTrue(np.array_equal(r[1:3], np.array([1,2])))
        self.assertTrue(np.array_equal(r[5:], np.array([5,6,7,8,9])))
        self.assertTrue(np.array_equal(r[5:100], np.array([5,6,7,8,9])))
        self.assertTrue(np.array_equal(r[[1,2,3]], np.array([1,2,3])))
        self.assertTrue(np.array_equal(r[np.array([1,2,3])], np.array([1,2,3])))
        i = iter([1,2,3])
        self.assertTrue(np.array_equal(r[i], np.array([1,2,3])))

        with self.assertRaises(RuntimeError):
            r[3] = 4
        with self.assertRaises(IndexError):
            _ = r[10]
        with self.assertRaises(IndexError):
            _ = r[[8,9,10]]
        with self.assertRaises(IndexError):
            _ = r[np.array([8,9,10])]

        # should not raise anything
        r[9] = 9 
        r[4:5] = [4]
        r[[1,2,3]] = [1,2,3]
        r[[1,2,3]] = np.array([1,2,3])
        with self.assertRaises(RangeError):
            r[9] = 8
        with self.assertRaises(RangeError):
            r[4:5] = [5]
        with self.assertRaises(RangeError):
            r[[4,5,6]] = [4,5,7]
        for i, j in zip(range(10), r):
            self.assertEqual(i, j)
