from . import ProgressiveTest

from progressivis import Scheduler
from progressivis.table.table import Table

import numpy as np
import pandas as pd

from typing import Any


class TestTableDel(ProgressiveTest):
    def setUp(self) -> None:
        super().setUp()
        self.scheduler_ = Scheduler.default

    def test_del(self) -> None:
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

    def test_del2(self) -> None:
        t = Table("table_filtering", dshape="{a: int, b: float32}", create=True)
        sz = 20
        sz_del = 10
        sz_add = 5
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
        ivalues2: np.ndarray[Any, Any] = np.random.randint(100, size=sz_add)
        fvalues2: np.ndarray[Any, Any] = np.random.rand(sz_add) * 100
        dict_add = {"a": ivalues2, "b": fvalues2}
        ix = range(df.index[-1] + 1, df.index[-1] + 1 + sz_add)
        df = pd.concat([df, pd.DataFrame(dict_add, index=ix)])
        t.append(data=dict_add)
        self.assertSetEqual(set(t.index), set(df.index))

    def test_del3(self) -> None:
        t = Table("table_filtering", dshape="{a: int, b: float32}", create=True)
        sz = 20
        sz_del = 10
        sz_add = 15
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
        ivalues2: np.ndarray[Any, Any] = np.random.randint(100, size=sz_add)
        fvalues2: np.ndarray[Any, Any] = np.random.rand(sz_add) * 100
        dict_add = {"a": ivalues2, "b": fvalues2}
        ix = range(df.index[-1] + 1, df.index[-1] + 1 + sz_add)
        df = pd.concat([df, pd.DataFrame(dict_add, index=ix)])
        t.append(data=dict_add)
        self.assertSetEqual(set(t.index), set(df.index))
