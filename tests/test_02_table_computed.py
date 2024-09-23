from . import ProgressiveTest

from progressivis import Scheduler
from progressivis.table.table import PTable
from progressivis.table.table_base import BasePTable
from progressivis.core.pintset import PIntSet
from progressivis.table.compute import week_day, SingleColFunc, MultiColFunc, MultiColExpr
import numpy as np
from typing import Any, Dict


class TestPTableSelected(ProgressiveTest):
    def setUp(self) -> None:
        super(TestPTableSelected, self).setUp()
        self.scheduler_ = Scheduler.default

    def test_loc_table_computed(self) -> None:
        t = PTable(
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
        colfunc = SingleColFunc(func=np.arcsin, base="b")
        t.computed["arcsin_b"] = colfunc
        self.assertEqual(t.shape, (20, 2))
        tb = t.loc[:, "b"]
        assert tb
        self.assertEqual(tb.shape, (20, 1))
        tab = t.loc[:, "arcsin_b"]
        assert tab
        self.assertEqual(tab.shape, (20, 1))
        sel = PIntSet(range(5, 8))
        view = t.loc[sel, :]
        assert view is not None
        self.assertEqual(view.shape, (3, 3))
        view2 = view.loc[sel, ["b", "arcsin_b"]]
        assert view2 is not None
        self.assertEqual(view2.shape, (3, 2))
        self.assertTrue(np.allclose(np.arcsin(tb.to_array()), tab.to_array()))
        self.assertEqual(type(view), BasePTable)
        self.assertEqual(type(view2), BasePTable)
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

    def test_loc_table_computed2(self) -> None:
        t = PTable(
            "table_for_test_computed_columns2",
            dshape="{a: 6*uint16, b: float32}",
            create=True,
        )
        t.resize(10)
        sz = 20
        years = np.random.randint(2015, 2020, size=sz, dtype="uint16")
        months = np.random.randint(1, 12, size=sz, dtype="uint16")
        days = np.random.randint(1, 28, size=sz, dtype="uint16")
        hours = np.random.randint(0, 23, size=sz, dtype="uint16")
        mins = np.random.randint(0, 59, size=sz, dtype="uint16")
        secs = np.random.randint(0, 59, size=sz, dtype="uint16")
        dt_values = np.vstack([years, months, days, hours, mins, secs]).T
        t["a"] = dt_values[:10]
        fvalues = np.array(np.random.rand(20), np.float32)
        t["b"] = fvalues[:10]
        self.assertEqual(t.shape, (10, 7))
        t.append({"a": dt_values[10:], "b": fvalues[10:]})
        self.assertEqual(t.shape, (20, 7))
        colfunc = SingleColFunc(func=week_day, base="a", dtype=np.dtype("O"))
        t.computed["weekday"] = colfunc
        self.assertEqual(t.shape, (20, 7))
        ta = t.loc[:, "a"]
        assert ta
        self.assertEqual(ta.shape, (20, 6))
        tdw = t.loc[:, "weekday"]
        assert tdw
        self.assertEqual(tdw.shape, (20, 1))
        sel = PIntSet(range(5, 8))
        view = t.loc[sel, :]
        assert view is not None
        self.assertEqual(view.shape, (3, 8))
        view2 = view.loc[sel, ["a", "weekday"]]
        assert view2 is not None
        self.assertEqual(view2.shape, (3, 7))
        self.assertTrue(
            np.array_equal(
                np.array([week_day(v) for v in ta["a"].loc[:]]),
                tdw.to_array().reshape(-1),
            )
        )

    def test_loc_table_computed_numexpr(self) -> None:
        t = PTable(
            "table_for_test_computed_columns_ne",
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
        colexpr = MultiColExpr(expr="a*b", base=["a", "b"], dtype=np.dtype("float32"))
        t.computed["a_x_b"] = colexpr
        self.assertEqual(t.shape, (20, 2))
        ta = t.loc[:, "a"]
        tb = t.loc[:, "b"]
        assert ta
        assert tb
        self.assertEqual(ta.shape, (20, 1))
        self.assertEqual(tb.shape, (20, 1))
        taxb = t.loc[:, "a_x_b"]
        assert taxb
        self.assertEqual(taxb.shape, (20, 1))
        sel = PIntSet(range(5, 8))
        view = t.loc[sel, :]
        assert view is not None
        self.assertEqual(view.shape, (3, 3))
        view2 = view.loc[sel, ["b", "a_x_b"]]
        assert view2 is not None
        self.assertEqual(view2.shape, (3, 2))
        self.assertTrue(np.allclose(ta.to_array() * tb.to_array(), taxb.to_array()))
        self.assertEqual(type(view), BasePTable)
        self.assertEqual(type(view2), BasePTable)
        self.assertTrue(np.array_equal(view[0].value, ivalues[5:8]))
        self.assertTrue(np.array_equal(view[1].value, fvalues[5:8]))
        self.assertTrue(np.allclose(view[2].value, ivalues[5:8] * fvalues[5:8]))
        self.assertEqual(view.at[6, "a"], ivalues[6])
        self.assertEqual(view.at[7, "b"], fvalues[7])
        self.assertEqual(view.at[7, "a_x_b"], np.float32(ivalues[7] * fvalues[7]))
        with self.assertRaises(KeyError):
            self.assertEqual(view.at[4, "a"], ivalues[4])
        with self.assertRaises(KeyError):
            self.assertEqual(view.at[8, "a"], ivalues[8])

    def test_loc_table_computed_vect_func(self) -> None:
        t = PTable(
            "table_for_test_computed_columns_vf",
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

        def _axb(index: Any, local_dict: Dict[str, Any]) -> Any:
            return local_dict["a"] * local_dict["b"]

        colfunc = MultiColFunc(func=_axb, base=["a", "b"], dtype=np.dtype("float32"))
        t.computed["a_x_b"] = colfunc
        self.assertEqual(t.shape, (20, 2))
        ta = t.loc[:, "a"]
        tb = t.loc[:, "b"]
        assert ta
        assert tb
        self.assertEqual(ta.shape, (20, 1))
        self.assertEqual(tb.shape, (20, 1))
        taxb = t.loc[:, "a_x_b"]
        assert taxb
        self.assertEqual(taxb.shape, (20, 1))
        sel = PIntSet(range(5, 8))
        view = t.loc[sel, :]
        assert view is not None
        self.assertEqual(view.shape, (3, 3))
        view2 = view.loc[sel, ["b", "a_x_b"]]
        assert view2 is not None
        self.assertEqual(view2.shape, (3, 2))
        self.assertTrue(np.allclose(ta.to_array() * tb.to_array(), taxb.to_array()))
        self.assertEqual(type(view), BasePTable)
        self.assertEqual(type(view2), BasePTable)
        self.assertTrue(np.array_equal(view[0].value, ivalues[5:8]))
        self.assertTrue(np.array_equal(view[1].value, fvalues[5:8]))
        self.assertTrue(np.allclose(view[2].value, ivalues[5:8] * fvalues[5:8]))
        self.assertEqual(view.at[6, "a"], ivalues[6])
        self.assertEqual(view.at[7, "b"], fvalues[7])
        self.assertEqual(view.at[7, "a_x_b"], ivalues[7] * fvalues[7])
        with self.assertRaises(KeyError):
            self.assertEqual(view.at[4, "a"], ivalues[4])
        with self.assertRaises(KeyError):
            self.assertEqual(view.at[8, "a"], ivalues[8])
