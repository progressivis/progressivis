from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print
from progressivis.linalg.linear_map import LinearMap
import numpy as np
from progressivis.stats import RandomTable


class TestLinearMap(ProgressiveTest):
    def test_linear_map(self) -> None:
        s = self.scheduler()
        vectors = RandomTable(3, rows=100000, scheduler=s)
        transf = RandomTable(10, rows=3, scheduler=s)
        module = LinearMap(scheduler=s)
        module.input.vectors = vectors.output.result
        module.input.transformation = transf.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.matmul(vectors.table.to_array(), transf.table.to_array())
        res2 = module.table.to_array()
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_linear_map2(self) -> None:
        s = self.scheduler()
        vectors = RandomTable(20, rows=100000, scheduler=s)
        transf = RandomTable(20, rows=3, scheduler=s)
        module = LinearMap(columns=["_3", "_4", "_5"], scheduler=s)
        module.input.vectors = vectors.output.result
        module.input.transformation = transf.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.matmul(vectors.table.to_array()[:, 2:5], transf.table.to_array())
        res2 = module.table.to_array()
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_linear_map3(self) -> None:
        s = self.scheduler()
        vectors = RandomTable(20, rows=100000, scheduler=s)
        transf = RandomTable(20, rows=3, scheduler=s)
        module = LinearMap(
            columns=["_3", "_4", "_5"],
            transf_columns=["_4", "_5", "_6", "_7"],
            scheduler=s,
        )
        module.input.vectors = vectors.output.result
        module.input.transformation = transf.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.matmul(
            vectors.table.to_array()[:, 2:5], transf.table.to_array()[:, 3:7]
        )
        res2 = module.table.to_array()
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))
