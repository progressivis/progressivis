from . import ProgressiveTest, skip
import pandas as pd
from progressivis import Print
from progressivis.stats.scaling import MinMaxScaler
from progressivis.core import aio
from progressivis.io import SimpleCSVLoader
from progressivis.table.constant import ConstDict
from progressivis.utils.psdict import PDict

import numpy as np
import tempfile as tf
import os

np.random.seed(1)
N_ROWS = 100_000
df = pd.DataFrame(
    {
        "A": np.random.normal(0, 3, N_ROWS),
        "B": np.random.normal(5, 2, N_ROWS),
        "C": np.random.normal(-5, 4, N_ROWS),
        "D": np.random.normal(5, 3, N_ROWS),
    }
)

df2 = pd.concat([df, df * 1.0])


class TestScalers(ProgressiveTest):
    @skip("not ready yet")
    def test_min_max_scaler(self) -> None:
        s = self.scheduler()
        _, f = tf.mkstemp()
        print(f)
        df.to_csv(f, index=False)
        cols = ["A", "B"]
        csv = SimpleCSVLoader(f, usecols=cols, scheduler=s)
        sc = MinMaxScaler(reset_threshold=10_000, scheduler=s)
        # sc.input[0] = random.output.result
        sc.create_dependent_modules(csv)
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = sc.output.result
        aio.run(s.start())
        assert sc.result is not None
        os.unlink(f)
        for c in cols:
            self.assertGreaterEqual(min(sc.result[c]), 0.0)
            self.assertLessEqual(max(sc.result[c]), 1.0)

    @skip("not ready yet")
    def test_min_max_scaler_tol(self) -> None:
        s = self.scheduler()
        _, f = tf.mkstemp()
        print(f)
        df2.to_csv(f, index=False)
        cols = ["A", "B"]
        csv = SimpleCSVLoader(f, usecols=cols, throttle=100, scheduler=s)
        cst = ConstDict(pdict=PDict({"delta": -5, "ignore_max": 10}), scheduler=s)
        sc = MinMaxScaler(reset_threshold=10_000, scheduler=s)
        # sc.input[0] = random.output.result
        sc.create_dependent_modules(csv)
        sc.input.control = cst.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr2 = Print(proc=self.terse, scheduler=s)
        pr.input[0] = sc.output.result
        pr2.input[0] = sc.output.info
        aio.run(s.start())
        assert csv.result is not None
        assert sc.result is not None
        assert sc.info is not None
        print(sc.info)
        os.unlink(f)
        self.assertEqual(len(csv.result) - sc.info["ignored"], len(sc.result))
        for c in cols:
            self.assertGreaterEqual(min(sc.result[c]), 0.0)
            self.assertLessEqual(max(sc.result[c]), 1.0)
