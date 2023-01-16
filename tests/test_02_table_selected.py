from . import ProgressiveTest

from progressivis import Scheduler
from progressivis.table.table import PTable
from progressivis.table.table_base import BasePTable
from progressivis.core.pintset import PIntSet

import numpy as np


class TestPTableSelected(ProgressiveTest):
    def setUp(self) -> None:
        super(TestPTableSelected, self).setUp()
        self.scheduler_ = Scheduler.default

    def test_loc_table_selection(self) -> None:
        t = PTable("table_for_sel", dshape="{a: int, b: float32}", create=True)
        t.resize(10)
        ivalues = np.random.randint(100, size=20)
        t["a"] = ivalues[:10]
        fvalues = np.array(np.random.rand(20), np.float32)
        t["b"] = fvalues[:10]
        t.append({"a": ivalues[10:], "b": fvalues[10:]})
        sel = PIntSet(range(5, 8))
        view = t.loc[sel, :]
        assert view is not None
        self.assertEqual(type(view), BasePTable)
        self.assertTrue(np.array_equal(view[0].value, ivalues[5:8]))
        self.assertEqual(view.at[6, "a"], ivalues[6])
        self.assertEqual(view.at[7, "b"], fvalues[7])
        with self.assertRaises(KeyError):
            self.assertEqual(view.at[4, "a"], ivalues[4])
        with self.assertRaises(KeyError):
            self.assertEqual(view.at[8, "a"], ivalues[8])
