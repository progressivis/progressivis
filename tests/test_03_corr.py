from . import ProgressiveTest
import pandas as pd
from progressivis import Print
from progressivis.stats import Var, VarH, RandomTable
from progressivis.stats.correlation import OnlineCovariance, Cov, Corr
from progressivis.datasets import get_dataset
from progressivis.core import aio
from sklearn.utils import gen_batches
import numpy as np

class TestCorr(ProgressiveTest):

    def test_online_cov(self):
        s = self.scheduler()
        random = RandomTable(2, rows=100_000, scheduler=s)
        cov = Cov(scheduler=s)
        cov.input[0] = random.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = cov.output.result
        aio.run(s.start())
        res1 = np.cov(random.result.to_array().T)
        res2 = cov.result_as_df(['_1', '_2']).values
        self.assertTrue(np.allclose(res1, res2))

    def test_online_corr(self):
        s = self.scheduler()
        random = RandomTable(2, rows=100_000, scheduler=s)
        corr = Corr(scheduler=s)
        corr.create_dependent_modules(random)
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = corr.output.result
        aio.run(s.start())
        cols = ['_1', '_2']
        arr = random.result.to_array()
        res1 = pd.DataFrame(arr, columns=cols, dtype='float64').corr()
        res2 = corr.result_as_df(cols).values
        self.assertTrue(np.allclose(res1, res2))
