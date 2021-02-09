from . import ProgressiveTest
import copy
from progressivis.core.bitmap import bitmap
from progressivis.utils.psdict import PsDict
import numpy as np


class TestPsDict(ProgressiveTest):
    def test_init_dict(self):
        d1 = PsDict(a=1, b=2, c=3)
        other = dict(a=1, b=2, c=3)
        d2 = PsDict(other)
        self.assertEqual(d1, d2)
        with self.assertRaises(TypeError):
            _ = PsDict(other, **other)
        d3 = PsDict(other, x=8, y=5)
        self.assertEqual(len(d3), 5)

    def test_ps_dict_new_ids(self):
        prev = PsDict(a=1, b=2, c=3)
        now = copy.copy(prev)
        now['x'] = 10
        now['y'] = 20
        new_ids = now.new_indices(prev)
        self.assertEqual(bitmap(new_ids), bitmap([3, 4]))

    def test_ps_dict_updated_ids(self):
        prev = PsDict(a=1, b=2, c=3, d=4, e=5)
        now = copy.copy(prev)
        updated_ids = now.updated_indices(prev)
        self.assertEqual(bitmap(updated_ids), bitmap())
        now['b'] += 1
        now['d'] *= 2
        updated_ids = now.updated_indices(prev)
        self.assertEqual(bitmap(updated_ids), bitmap([1, 3]))

    def test_ps_dict_deleted_ids(self):
        prev = PsDict(a=1, b=2, c=3, d=4, e=5)
        now = copy.copy(prev)
        deleted_ids = now.deleted_indices(prev)
        self.assertEqual(bitmap(deleted_ids), bitmap())
        del now['b']
        now['c'] *= 3
        deleted_ids = now.deleted_indices(prev)
        updated_ids = now.updated_indices(prev)
        self.assertEqual(bitmap(deleted_ids), bitmap([1]))
        self.assertEqual(bitmap(updated_ids), bitmap([2]))
