from . import ProgressiveTest
import copy
from progressivis.core.pintset import PIntSet
from progressivis.utils.psdict import PDict


class TestPDict(ProgressiveTest):
    def test_init_dict(self) -> None:
        d1 = PDict(a=1, b=2, c=3)
        other = dict(a=1, b=2, c=3)
        d2 = PDict(other)
        self.assertEqual(d1, d2)
        d3 = PDict(other, x=8, y=5)
        self.assertEqual(len(d3), 5)

    def test_ps_dict_new_ids(self) -> None:
        prev = PDict(a=1, b=2, c=3)
        now = copy.copy(prev)
        now["x"] = 10
        now["y"] = 20
        new_ids = now.created_indices(prev)
        self.assertEqual(PIntSet(new_ids), PIntSet([3, 4]))

    def test_ps_dict_updated_ids(self) -> None:
        prev = PDict(a=1, b=2, c=3, d=4, e=5)
        now = copy.copy(prev)
        updated_ids = now.updated_indices(prev)
        self.assertEqual(PIntSet(updated_ids), PIntSet())
        now["b"] += 1
        now["d"] *= 2
        updated_ids = now.updated_indices(prev)
        self.assertEqual(PIntSet(updated_ids), PIntSet([1, 3]))

    def test_ps_dict_deleted_ids(self) -> None:
        prev = PDict(a=1, b=2, c=3, d=4, e=5)
        now = copy.copy(prev)
        deleted_ids = now.deleted_indices(prev)
        self.assertEqual(PIntSet(deleted_ids), PIntSet())
        del now["b"]
        now["c"] *= 3
        deleted_ids = now.deleted_indices(prev)
        updated_ids = now.updated_indices(prev)
        self.assertEqual(PIntSet(deleted_ids), PIntSet([1]))
        self.assertEqual(PIntSet(updated_ids), PIntSet([2]))
