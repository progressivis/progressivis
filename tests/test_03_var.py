from . import ProgressiveTest

from progressivis import Print
from progressivis.stats import Var, RandomTable
from progressivis.core import aio
import numpy as np


class Testvar(ProgressiveTest):
    def test_var(self):
        s = self.scheduler()
        random = RandomTable(1, rows=1000, scheduler=s)
        var=Var(scheduler=s)
        var.input.table = random.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = var.output.result
        aio.run(s.start())
        res1 = np.array([float(e) for e in random.result.var(ddof=1).values()])
        res2 = np.array([float(e) for e in var.result.last().to_dict(ordered=True).values()])
        print('res1:', res1)
        print('res2:', res2)
        self.assertTrue(np.allclose(res1, res2))

if __name__ == '__main__':
    ProgressiveTest.main()
