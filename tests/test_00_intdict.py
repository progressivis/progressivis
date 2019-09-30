from . import ProgressiveTest

from progressivis.utils.intdict import IntDict
import numpy as np


class TestIntDict(ProgressiveTest):
    def test_intdict(self):
        d = IntDict()
        self.assertEqual(len(d), 0)
        with self.assertRaises(KeyError):
            x = d[10]
            #print(x)
        keys = np.arange(10, dtype=np.int64)
        values = keys+10
        d = IntDict(keys, values)
        self.assertEqual(len(d), 10)
        with self.assertRaises(KeyError):
            x = d[11]
            #print(x)
        for (k, v) in zip(keys, values):
            self.assertEqual(d[k], v)

        del d[5]
        self.assertEqual(len(d), 9)
        with self.assertRaises(KeyError):
            x = d[5]
            #print(x)

        d[5] = 15 # put it back
        d.get_values(keys) # overrides keys
        self.assertTrue((keys==values).all())
        
        keys = np.arange(10, 20, dtype=np.int64)
        values = keys+10
        d.update(keys, values)
        
        self.assertEqual(len(d), 20)
        with self.assertRaises(KeyError):
            x = d[21]
            #print(x)
        for (k, v) in zip(keys, values):
            self.assertEqual(d[k], v)
        
        rk = np.random.choice(np.arange(20, dtype=np.int64), 5)
        rv = rk+10
        d.get_items(rk)
        self.assertTrue((rk==rv).all())
        
        self.assertTrue(10 in d)
        self.assertFalse(22 in d)

        self.assertTrue(d.contains_any(np.arange(19,21, dtype=np.int64)))
        self.assertFalse(d.contains_any(np.arange(20, 22, dtype=np.int64)))
