from __future__ import annotations

from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print
from progressivis.stats import RandomPTable
from progressivis.table.module import PTableModule
from progressivis.table.table import PTable
from progressivis.core.slot import SlotDescriptor
from progressivis.core.utils import fix_loc
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.module import ReturnRunStep


class Hadamard(PTableModule):
    inputs = [
        SlotDescriptor("x1", type=PTable, required=True),
        SlotDescriptor("x2", type=PTable, required=True),
    ]

    def reset(self) -> None:
        if self.result is not None:
            self.table.resize(0)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        x1 = self.get_input_slot("x1")
        x2 = self.get_input_slot("x2")
        if x1.updated.any() or x1.deleted.any() or x2.updated.any() or x2.deleted.any():
            x1.reset()
            x2.reset()
            if self.result is not None:
                self.table.resize(0)
            x1.update(run_number)
            x2.update(run_number)
        step_size = min(x1.created.length(), x2.created.length(), step_size)
        x1_indices = x1.created.next(step_size)
        x2_indices = x2.created.next(step_size)
        res = {}
        data1 = x1.data().loc[fix_loc(x1_indices)]
        data2 = x2.data().loc[fix_loc(x2_indices)]
        assert data1.columns == data2.columns
        for col in data1.columns:
            res[col] = np.multiply(data1[col].value, data2[col].value)
        if self.result is None:
            self.result = PTable(name="simple_hadamard", data=res, create=True)
        else:
            self.table.append(res)
        return self._return_run_step(self.next_state(x1), steps_run=step_size)


class TestHadamard(ProgressiveTest):
    def test_hadamard(self) -> None:
        s = self.scheduler()
        random1 = RandomPTable(3, rows=100000, scheduler=s)
        random2 = RandomPTable(3, rows=100000, scheduler=s)
        module = Hadamard(scheduler=s)
        module.input.x1 = random1.output.result
        module.input.x2 = random2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.result
        aio.run(s.start())
        res1 = np.multiply(random1.table.to_array(), random2.table.to_array())
        res2 = module.table.to_array()
        self.assertTrue(np.allclose(res1, res2, equal_nan=True))


if __name__ == "__main__":
    ProgressiveTest.main()
