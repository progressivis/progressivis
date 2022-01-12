from __future__ import annotations

import numpy as np
import pandas as pd

from . import ProgressiveTest, skip

from progressivis import Scheduler
from progressivis.table.table import Table


from typing import Any, cast


class TestTableEval(ProgressiveTest):
    def setUp(self) -> None:
        super(TestTableEval, self).setUp()
        self.scheduler_ = Scheduler.default

    def test_filtering(self) -> None:
        t = Table("table_filtering", dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100, size=20)
        t["a"] = ivalues
        fvalues: np.ndarray[Any, Any] = np.random.rand(20) * 100
        t["b"] = fvalues
        df = pd.DataFrame(t.to_dict())

        def small_fun(expr: str, r: Any) -> None:
            te = t.eval(expr, result_object=r)
            dfe = df.eval(expr)
            self.assertTrue(np.array_equal(te["a"].loc[:], df[dfe]["a"]))
            self.assertTrue(np.allclose(te["b"].loc[:], df[dfe]["b"]))

        def small_fun_ne(expr: str) -> None:
            r = "raw_numexpr"
            te = t.eval(expr, result_object=r)
            dfe: pd.PandasObject = df.eval(expr)
            self.assertTrue(np.array_equal(te, dfe.values))

        small_fun_ne("(a>10) & (a <80)")
        small_fun_ne("(b>10) & (b <80)")
        small_fun_ne("a>=b")
        small_fun("(a>10) & (a <80)", "table")
        small_fun("(b>10) & (b <80)", "table")
        small_fun("a>=b", "table")
        small_fun("(a>10) & (a <80)", "view")

    def test_filtering2(self) -> None:
        t = Table("table_filtering", dshape="{a: int, b: float32}", create=True)
        sz = 1000
        sz_del = 100
        t.resize(sz)
        np.random.seed(42)
        ivalues = np.random.randint(100, size=sz)
        t["a"] = ivalues
        fvalues: np.ndarray[Any, Any] = np.random.rand(sz) * 100
        t["b"] = fvalues
        df = pd.DataFrame(t.to_dict())
        to_del = np.random.randint(len(t) - 1, size=sz_del)
        del t.loc[to_del]
        df = df.drop(to_del)
        self.assertListEqual(list(t.index), list(df.index))

        def small_fun_index(expr: str) -> None:
            ix = t.eval(expr)
            dfe = df.eval(expr)
            self.assertSetEqual(set(ix), set(df.index[dfe]))

        small_fun_index("(a>10) & (a <80)")

    def test_assign(self) -> None:
        t = Table("table_eval_assign", dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100, size=20)
        t["a"] = ivalues
        fvalues: np.ndarray[Any, Any] = np.random.rand(20) * 100
        t["b"] = fvalues
        df = pd.DataFrame(t.to_dict())
        t2 = t.eval("a = a+2*b", inplace=False)
        df2 = cast(pd.DataFrame, df.eval("a = a+2*b", inplace=False))
        self.assertTrue(np.allclose(t2["a"], df2["a"]))
        self.assertTrue(np.allclose(t2["b"], df2["b"]))
        t.eval("b = a+2*b", inplace=True)
        df.eval("b = a+2*b", inplace=True)
        self.assertTrue(np.allclose(t["a"].values, df["a"].values))
        self.assertTrue(np.allclose(t["b"].values, df["b"].values))

    @skip("Not Ready")
    def test_user_dict(self) -> None:
        t = Table("table_user_dict", dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100, size=20)
        t["a"] = ivalues
        fvalues: np.ndarray[Any, Any] = np.random.rand(20) * 100
        t["b"] = fvalues
        df = pd.DataFrame(t.to_dict())
        _ = t.eval("a = a+2*b", inplace=False)
        _ = df.eval("x = a.loc[3]+2*b.loc[3]", inplace=False)
        # print(df2.x)
        # self.assertTrue(np.allclose(t2['a'], df2['a']))
        # self.assertTrue(np.allclose(t2['b'], df2['b']))
        # t.eval('b = a+2*b', inplace=True)
        # df.eval('b = a+2*b', inplace=True)
        # self.assertTrue(np.allclose(t['a'], df['a']))
        # self.assertTrue(np.allclose(t['b'], df['b']))
