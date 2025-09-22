from . import ProgressiveTest

from progressivis import RandomPTable, Sink
from progressivis.table.repeater import Repeater, Computed
from progressivis.core import aio

import numpy as np


class TestRepeater(ProgressiveTest):
    def test_repeater(self) -> None:
        comp = Computed()
        func = np.log
        comp.add_ufunc_column("log_1", "_1", func, np.dtype("float64"))
        s = self.scheduler
        random = RandomPTable(10, rows=10_000, scheduler=s)
        rep = Repeater(computed=comp, scheduler=s)
        rep.input.table = random.output.result["_2", "_3"]
        sink = Sink(scheduler=s)
        sink.input.inp = rep.output.result
        aio.run(s.start())
        assert random.result is not None
        assert rep.result is not None
        res1 = np.log(random.result["_1"].loc[:])
        res2 = rep.result["log_1"].loc[:]
        self.assertTrue(np.allclose(res1, res2))
