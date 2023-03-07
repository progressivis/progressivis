from . import ProgressiveTest
import numpy as np
from progressivis import Print
from progressivis.stats import IdxMax, IdxMin, Max, Min, RandomPTable
from progressivis.table.stirrer import Stirrer
from progressivis.core import aio, notNone

from typing import Any, Dict


class TestIdxMax(ProgressiveTest):
    def tearDown(self) -> None:
        TestIdxMax.cleanup()

    def test_idxmax(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10000, throttle=1000, scheduler=s)
        idxmax = IdxMax(scheduler=s)
        idxmax.input[0] = random.output.result
        max_ = Max(scheduler=s)
        max_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = idxmax.output.result
        pr2 = Print(proc=self.terse, scheduler=s)
        pr2.input[0] = max_.output.result
        aio.run(s.start())
        max1 = max_.result
        assert max1 is not None
        max = idxmax._max
        assert max is not None
        max2 = notNone(max.last()).to_dict()
        self.compare(max1, max2)

    def test_idxmax2(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10000, throttle=1000, scheduler=s)
        stirrer = Stirrer(
            update_column="_1", delete_rows=5, fixed_step_size=100, scheduler=s
        )
        stirrer.input[0] = random.output.result
        idxmax = IdxMax(scheduler=s)
        idxmax.input[0] = stirrer.output.result
        max_ = Max(scheduler=s)
        max_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = idxmax.output.result
        pr2 = Print(proc=self.terse, scheduler=s)
        pr2.input[0] = max_.output.result
        aio.run(s.start())
        max1 = max_.result
        assert max1 is not None
        max = idxmax._max
        assert max is not None
        max2 = notNone(max.last()).to_dict()
        self.compare(max1, max2)

    def test_idxmin(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10000, throttle=1000, scheduler=s)
        idxmin = IdxMin(scheduler=s)
        idxmin.input[0] = random.output.result
        min_ = Min(scheduler=s)
        min_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = idxmin.output.result
        pr2 = Print(proc=self.terse, scheduler=s)
        pr2.input[0] = min_.output.result
        aio.run(s.start())
        min1 = min_.result
        assert min1 is not None
        min = idxmin._min
        assert min is not None
        min2 = notNone(min.last()).to_dict()
        self.compare(min1, min2)

    def test_idxmin2(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10000, throttle=1000, scheduler=s)
        stirrer = Stirrer(
            update_column="_1", delete_rows=5, fixed_step_size=100, scheduler=s
        )
        stirrer.input[0] = random.output.result
        idxmin = IdxMin(scheduler=s)
        idxmin.input[0] = stirrer.output.result
        min_ = Min(scheduler=s)
        min_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = idxmin.output.result
        pr2 = Print(proc=self.terse, scheduler=s)
        pr2.input[0] = min_.output.result
        aio.run(s.start())
        min1 = min_.result
        assert min1 is not None
        min = idxmin._min
        assert min is not None
        min2 = notNone(min.last()).to_dict()
        self.compare(min1, min2)

    def compare(self, res1: Dict[str, Any], res2: Dict[str, Any]) -> None:
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        self.assertTrue(np.allclose(v1, v2))


if __name__ == "__main__":
    ProgressiveTest.main()
