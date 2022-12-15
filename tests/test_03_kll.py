from . import ProgressiveTest, skipIf
import os
from progressivis.core import aio
from progressivis import Print
from progressivis.stats.kll import KLLSketch
from progressivis.stats import RandomTable
import numpy as np
from datasketches import kll_floats_sketch

K = 300
BINS = 128
QUANTILES = [0.3, 0.5, 0.7]
SPLITS_SEQ = [0.3, 0.5, 0.7]
SPLITS_DICT = dict(lower=0.1, upper=0.9, n_splits=10)


@skipIf(os.getenv("CI"), "randomly fails on CI")
class TestKll(ProgressiveTest):
    def test_kll(self):
        np.random.seed(42)
        s = self.scheduler()
        random = RandomTable(3, rows=10_000, scheduler=s)
        kll = KLLSketch(column="_1", scheduler=s)
        kll.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = kll.output.result
        aio.run(s.start())
        val = random.result["_1"].value
        sk = kll_floats_sketch(K)
        sk.update(val)
        self.assertAlmostEqual(kll.result["max"], sk.get_max_value())
        self.assertAlmostEqual(kll.result["min"], sk.get_min_value())
        self.assertEqual(kll.result["quantiles"], [])
        self.assertEqual(kll.result["splits"], [])
        self.assertEqual(kll.result["pmf"], [])

    def test_kll2(self):
        np.random.seed(42)
        s = self.scheduler()
        random = RandomTable(3, rows=10_000, scheduler=s)
        kll = KLLSketch(column="_1", scheduler=s)
        kll.params.quantiles = QUANTILES
        kll.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = kll.output.result
        aio.run(s.start())
        val = random.result["_1"].value
        sk = kll_floats_sketch(K)
        sk.update(val)
        self.compare(kll.result["quantiles"], sk.get_quantiles(QUANTILES))

    def test_kll3(self):
        np.random.seed(42)
        s = self.scheduler()
        random = RandomTable(3, rows=10_000, scheduler=s)
        kll = KLLSketch(column="_1", scheduler=s)
        kll.params.binning = BINS
        kll.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = kll.output.result
        aio.run(s.start())
        val = random.result["_1"].value
        sk = kll_floats_sketch(K)
        sk.update(val)
        max_ = sk.get_max_value()
        min_ = sk.get_min_value()
        num_splits = BINS
        splits = np.linspace(min_, max_, num_splits)
        pmf = sk.get_pmf(splits[:-1])
        self.compare(kll.result["pmf"], pmf)

    def test_kll4(self):
        np.random.seed(42)
        s = self.scheduler()
        random = RandomTable(3, rows=10_000, scheduler=s)
        kll = KLLSketch(column="_1", scheduler=s)
        kll.params.binning = SPLITS_SEQ
        kll.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = kll.output.result
        aio.run(s.start())
        val = random.result["_1"].value
        sk = kll_floats_sketch(K)
        sk.update(val)
        pmf = sk.get_pmf(SPLITS_SEQ)
        self.compare(kll.result["pmf"], pmf)

    def test_kll5(self):
        np.random.seed(42)
        s = self.scheduler()
        random = RandomTable(3, rows=10_000, scheduler=s)
        kll = KLLSketch(column="_1", scheduler=s)
        kll.params.binning = SPLITS_DICT
        kll.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = kll.output.result
        aio.run(s.start())
        val = random.result["_1"].value
        sk = kll_floats_sketch(K)
        sk.update(val)
        lower_ = SPLITS_DICT["lower"]
        upper_ = SPLITS_DICT["upper"]
        num_splits = SPLITS_DICT["n_splits"]
        splits = np.linspace(lower_, upper_, num_splits)
        pmf = sk.get_pmf(splits[:-1])
        self.compare(kll.result["pmf"], pmf)

    def compare(self, res1, res2, atol=1e-02):
        v1 = np.array(res1)
        v2 = np.array(res2)
        self.assertEqual(v1.shape, v2.shape)
        self.assertTrue(np.allclose(v1, v2, atol=atol))


if __name__ == "__main__":
    ProgressiveTest.main()
