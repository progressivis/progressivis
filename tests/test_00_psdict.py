from . import ProgressiveTest

from progressivis.utils.psdict import FKSDict, PKDict
import numpy as np


class TestPsDict(ProgressiveTest):
    def test_init_dict(self):
        d1 = PKDict(a=1, b=2, c=3)
        other = dict(a=1, b=2, c=3)
        d2 = PKDict(other)
        self.assertEqual(d1, d2)
        with self.assertRaises(TypeError):
            _ = PKDict(other, **other)
        d3 = PKDict(other, x=8, y=5)
        self.assertEqual(len(d3), 5)

    def test_pk_dict(self):        
        d = PKDict(a=1, b=2, c=3)
        self.assertEqual(len(d), 3)
        with self.assertRaises(NotImplementedError):
            del d['b']
        d['b'] *= 2
        d['c'] += 1
        d['c'] -= 1
        self.assertEqual(d.activity['a'], 0)
        self.assertEqual(d.activity['b'], 1)
        self.assertEqual(d.activity['c'], 2)        

    def test_fks_dict(self):
        d = FKSDict(a=1, b=2, c=3)
        self.assertEqual(len(d), 3)
        with self.assertRaises(KeyError):
            d['x'] = 42
