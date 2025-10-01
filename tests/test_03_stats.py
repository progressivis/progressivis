from . import ProgressiveTest

from progressivis import Print, Stats, CSVLoader, Wait, get_dataset
from progressivis.core.api import notNone
from progressivis.core import aio

import numpy as np


class TestStats(ProgressiveTest):
    def test_stats(self) -> None:
        s = self.scheduler
        csv_module = CSVLoader(
            get_dataset("smallfile"), header=None, scheduler=s
        )
        stats = Stats("_1", name="test_stats", scheduler=s)
        wait = Wait(name="wait", delay=3, scheduler=s)
        wait.input.inp = csv_module.output.result
        stats.input._params = wait.output.out
        stats.input[0] = csv_module.output.result
        pr = Print(proc=self.terse, name="print", scheduler=s)
        pr.input[0] = stats.output.result
        aio.run(s.start())
        assert csv_module.result is not None
        assert stats.result is not None
        table = csv_module.result
        stable = stats.result
        last = notNone(stable.last())
        tmin = table["_1"].min()
        self.assertTrue(np.isclose(tmin, last["__1_min"]))
        tmax = table["_1"].max()
        self.assertTrue(np.isclose(tmax, last["__1_max"]))


if __name__ == "__main__":
    ProgressiveTest.main()
