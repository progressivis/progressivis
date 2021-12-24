from . import ProgressiveTest, skip

from progressivis import Scheduler
from progressivis.table.table import Table

import numpy as np
import pandas as pd


class TestTableEval(ProgressiveTest):
    def setUp(self):
        super(TestTableEval, self).setUp()
        self.scheduler = Scheduler.default

    def test_filtering(self):
        t = Table("table_filtering", dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100, size=20)
        t["a"] = ivalues
        fvalues = np.random.rand(20) * 100
        t["b"] = fvalues
        df = pd.DataFrame(t.to_dict())

        def small_fun(expr, r):
            te = t.eval(expr, result_object=r)
            dfe = df.eval(expr)
            self.assertTrue(np.array_equal(te["a"].loc[:], df[dfe]["a"]))
            self.assertTrue(np.allclose(te["b"].loc[:], df[dfe]["b"]))

        def small_fun_ne(expr):
            r = "raw_numexpr"
            te = t.eval(expr, result_object=r)
            dfe = df.eval(expr)
            self.assertTrue(np.array_equal(te, dfe.values))

        small_fun_ne("(a>10) & (a <80)")
        small_fun_ne("(b>10) & (b <80)")
        small_fun_ne("a>=b")
        small_fun("(a>10) & (a <80)", "table")
        small_fun("(b>10) & (b <80)", "table")
        small_fun("a>=b", "table")
        small_fun("(a>10) & (a <80)", "view")

    def test_filtering2(self):
        t = Table("table_filtering", dshape="{a: int, b: float32}", create=True)
        sz = 1000
        sz_del = 100
        t.resize(sz)
        np.random.seed(42)
        ivalues = np.random.randint(100, size=sz)
        t["a"] = ivalues
        fvalues = np.random.rand(sz) * 100
        t["b"] = fvalues
        df = pd.DataFrame(t.to_dict())
        to_del = np.random.randint(len(t) - 1, size=sz_del)
        del t.loc[to_del]
        df = df.drop(to_del)
        self.assertListEqual(list(t.index), list(df.index))

        def small_fun_index(expr):
            ix = t.eval(expr)
            dfe = df.eval(expr)
            self.assertSetEqual(set(ix), set(df.index[dfe]))

        small_fun_index("(a>10) & (a <80)")

    def test_assign(self):
        t = Table("table_eval_assign", dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100, size=20)
        t["a"] = ivalues
        fvalues = np.random.rand(20) * 100
        t["b"] = fvalues
        df = pd.DataFrame(t.to_dict())
        t2 = t.eval("a = a+2*b", inplace=False)
        df2 = df.eval("a = a+2*b", inplace=False)
        self.assertTrue(np.allclose(t2["a"], df2["a"]))
        self.assertTrue(np.allclose(t2["b"], df2["b"]))
        t.eval("b = a+2*b", inplace=True)
        df.eval("b = a+2*b", inplace=True)
        self.assertTrue(np.allclose(t["a"], df["a"]))
        self.assertTrue(np.allclose(t["b"], df["b"]))

    @skip("Not Ready")
    def test_user_dict(self):
        t = Table("table_user_dict", dshape="{a: int, b: float32}", create=True)
        t.resize(20)
        ivalues = np.random.randint(100, size=20)
        t["a"] = ivalues
        fvalues = np.random.rand(20) * 100
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
