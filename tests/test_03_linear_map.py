from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print, RandomPTable
from progressivis.linalg.linear_map import LinearMap
import numpy as np


class TestLinearMap(ProgressiveTest):
    def test_linear_map(self) -> None:
        s = self.scheduler
        vectors = RandomPTable(3, rows=100000, scheduler=s)
        transf = RandomPTable(10, rows=3, scheduler=s)
        module = LinearMap(scheduler=s)
        module.input.vectors = vectors.output.result
        module.input.transformation = transf.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        assert module.result is not None
        assert vectors.result is not None
        assert transf.result is not None
        res1 = np.matmul(vectors.result.to_array(), transf.result.to_array())
        res2 = module.result.to_array()
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_linear_map2(self) -> None:
        s = self.scheduler
        vectors = RandomPTable(20, rows=100000, scheduler=s)
        transf = RandomPTable(20, rows=3, scheduler=s)
        module = LinearMap(scheduler=s)
        module.input.vectors = vectors.output.result["_3", "_4", "_5"]
        module.input.transformation = transf.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        assert module.result is not None
        assert vectors.result is not None
        assert transf.result is not None
        res1 = np.matmul(vectors.result.to_array()[:, 2:5], transf.result.to_array())
        res2 = module.result.to_array()
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))

    def test_linear_map3(self) -> None:
        s = self.scheduler
        vectors = RandomPTable(20, rows=100000, scheduler=s)
        transf = RandomPTable(20, rows=3, scheduler=s)
        module = LinearMap(scheduler=s)
        module.input.vectors = vectors.output.result["_3", "_4", "_5"]
        module.input.transformation = transf.output.result["_4", "_5", "_6", "_7"]
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        assert module.result is not None
        assert vectors.result is not None
        assert transf.result is not None
        res1 = np.matmul(
            vectors.result.to_array()[:, 2:5], transf.result.to_array()[:, 3:7]
        )
        res2 = module.result.to_array()
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))
