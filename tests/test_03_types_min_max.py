from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print
from progressivis.stats import Min, Max
from progressivis.io import SimpleCSVLoader
from progressivis.datasets import get_dataset
import numpy as np


class TestMinMax(ProgressiveTest):
    def test_min(self):
        s = self.scheduler()
        random = SimpleCSVLoader(
            get_dataset("bigfile_multiscale"), nrows=10_000, scheduler=s
        )
        min_ = Min(name="min_" + str(hash(random)), scheduler=s)
        min_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        # s.join()
        res1 = random.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def compare(self, res1, res2):
        v1 = np.array(list(res1.values()), dtype=object)
        v2 = np.array(list(res2.values()), dtype=object)
        # print('v1 = ', v1)
        # print('v2 = ', v2)
        self.assertTrue(np.array_equal(v1, v2))

    def test_max(self):
        s = self.scheduler()
        random = SimpleCSVLoader(
            get_dataset("bigfile_multiscale"), nrows=10_000, scheduler=s
        )
        max_ = Max(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        # s.join()
        res1 = random.result.max()
        res2 = max_.result
        self.compare(res1, res2)


if __name__ == "__main__":
    ProgressiveTest.main()
