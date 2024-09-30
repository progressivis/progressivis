from . import ProgressiveTest

from progressivis import Print
from progressivis.stats import Var, VarH, RandomPTable
from progressivis.core.api import notNone
from progressivis.core import aio
import numpy as np


class Testvar(ProgressiveTest):
    def test_var_h(self) -> None:
        s = self.scheduler()
        random = RandomPTable(1, rows=1000, scheduler=s)
        var = VarH(scheduler=s)
        var.input.table = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = var.output.result
        aio.run(s.start())
        assert random.result is not None
        assert var.result is not None
        table = random.result
        assert table is not None
        res1 = [float(e) for e in table.var(ddof=1).values()]
        res2 = [float(e) for e in notNone(var.result.last()).to_dict(ordered=True).values()]
        self.assertTrue(np.allclose(res1, res2))

    def test_var(self) -> None:
        s = self.scheduler()
        random = RandomPTable(1, rows=1000, scheduler=s)
        var = Var(scheduler=s)
        var.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = var.output.result
        aio.run(s.start())
        assert random.result is not None
        assert var.result is not None
        res1 = [float(e) for e in random.result.var(ddof=1).values()]
        res2 = [float(e) for e in var.result.values()]
        self.assertTrue(np.allclose(res1, res2))


if __name__ == "__main__":
    ProgressiveTest.main()
