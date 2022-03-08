from . import ProgressiveTest, skip

from progressivis import Scheduler
from progressivis.table.table import Table
from progressivis.table.table_base import BaseTable
from progressivis.core.bitmap import bitmap

import numpy as np


class TestTableSelected(ProgressiveTest):
    def setUp(self) -> None:
        super(TestTableSelected, self).setUp()
        self.scheduler_ = Scheduler.default

    @skip
    def test_loc_table_computed(self) -> None:
        t = Table(
            "table_for_test_computed_columns",
            dshape="{a: int, b: float32}",
            create=True,
        )
        t.resize(10)
        ivalues = np.random.randint(100, size=20)
        t["a"] = ivalues[:10]
        fvalues = np.array(np.random.rand(20), np.float32)
        t["b"] = fvalues[:10]
        self.assertEqual(t.shape, (10, 2))
        t.append({"a": ivalues[10:], "b": fvalues[10:]})
        self.assertEqual(t.shape, (20, 2))
        t.add_computed("arcsin_b", "b", np.arcsin)
        self.assertEqual(t.shape, (20, 2))
        tb = t.loc[:, "b"]
        assert tb
        self.assertEqual(tb.shape, (20, 1))
        tab = t.loc[:, "arcsin_b"]
        assert tab
        self.assertEqual(tab.shape, (20, 1))
        sel = bitmap(range(5, 8))
        view = t.loc[sel, :]
        assert view is not None
        self.assertEqual(view.shape, (3, 3))
        view2 = view.loc[sel, ["b", "arcsin_b"]]
        assert view2 is not None
        self.assertEqual(view2.shape, (3, 2))
        self.assertTrue(np.allclose(np.arcsin(tb.to_array()), tab.to_array()))
        self.assertEqual(type(view), BaseTable)
        self.assertEqual(type(view2), BaseTable)
        self.assertTrue(np.array_equal(view[0].value, ivalues[5:8]))
        self.assertTrue(np.array_equal(view[1].value, fvalues[5:8]))
        self.assertTrue(np.array_equal(view[2].value, np.arcsin(fvalues[5:8])))
        self.assertEqual(view.at[6, "a"], ivalues[6])
        self.assertEqual(view.at[7, "b"], fvalues[7])
        self.assertEqual(view.at[7, "arcsin_b"], np.arcsin(fvalues[7]))
        with self.assertRaises(KeyError):
            self.assertEqual(view.at[4, "a"], ivalues[4])
        with self.assertRaises(KeyError):
            self.assertEqual(view.at[8, "a"], ivalues[8])
