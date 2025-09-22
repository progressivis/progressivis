from . import ProgressiveTest, skipIf

import os

import pandas as pd
from progressivis import Print, Corr, RandomPTable
from progressivis.core import aio
# from progressivis.stats import CovarianceMatrix

import numpy as np


class TestCorr(ProgressiveTest):
    def test_online_cov(self) -> None:
        s = self.scheduler
        random = RandomPTable(2, rows=100_000, scheduler=s)
        cov = Corr(mode="CovarianceOnly", scheduler=s)
        cov.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = cov.output.result
        aio.run(s.start())
        assert random.result is not None
        res1 = np.cov(random.result.to_array().T)
        res2 = cov.result_as_df(["_1", "_2"]).values
        self.assertTrue(np.allclose(res1, res2))

    def test_online_corr(self) -> None:
        s = self.scheduler
        random = RandomPTable(2, rows=100_000, scheduler=s)
        corr = Corr(scheduler=s)
        # corr.create_dependent_modules(random)
        corr.input.table = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = corr.output.result
        aio.run(s.start())
        assert random.result is not None
        cols = ["_1", "_2"]
        arr = random.result.to_array()
        res1 = pd.DataFrame(arr, columns=cols, dtype="float64").corr()
        res2 = corr.result_as_df(cols).values
        self.assertTrue(np.allclose(res1, res2))

    @skipIf(os.getenv("CI"), "Too long")
    def test_online_cov2(self) -> None:
        # def print_cov(d) -> None:
        #     print(CovarianceMatrix.as_pandas(d))

        s = self.scheduler
        random = RandomPTable(
            3,
            rows=1_000_000,
            throttle=10_000,
            scheduler=s
        )
        cov = Corr(scheduler=s)
        cov.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        # pr = Print(proc=print_cov, scheduler=s)
        pr.input[0] = cov.output.result
        aio.run(s.start())
        assert random.result is not None
        res1 = np.corrcoef(random.result.to_array().T)
        res2 = cov.result_as_df().values
        self.assertTrue(np.allclose(res1, res2))
