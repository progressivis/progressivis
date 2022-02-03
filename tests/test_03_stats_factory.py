from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print
from progressivis.vis import StatsFactory
from progressivis.stats import RandomTable
import numpy as np


class TestStatsFactory(ProgressiveTest):
    def test_sf(self):
        np.random.seed(42)
        s = self.scheduler()
        random = RandomTable(3, rows=10_000, scheduler=s)
        sf = StatsFactory(scheduler=s)
        sf.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = sf.output.result
        aio.run(s.start())
        print(s.modules())


if __name__ == "__main__":
    ProgressiveTest.main()
