from __future__ import annotations

import numpy as np

from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print, Min, RandomPTable, Module, def_input, def_output, PTable, process_slot, run_if_any, indices_len, fix_loc, PDict

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.module import ReturnRunStep


@def_input("table", PTable)
@def_output("result", PDict)
class Max(Module):
    """
    Simplified Max, adapted for documentation
    """

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.default_step_size = 10000

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.result.fill(-np.inf)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("table")
        if slot.updated.any() or slot.deleted.any():
            slot.reset()
            if self.result is not None:
                self.result.clear()  # resize(0)
            slot.update(run_number)
        indices = slot.created.next(step_size)
        steps = indices_len(indices)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        data = slot.data()
        op = data.loc[fix_loc(indices)].max(keepdims=False)
        if self.result is None:
            self.result = PDict(op)
        else:
            for k, v in self.result.items():
                self.result[k] = np.maximum(op[k], v)
        return self._return_run_step(self.next_state(slot), steps_run=steps)


@def_input("table", PTable)
@def_output("result", PDict)
class MaxDec(Module):
    """
    Simplified Max with decorated run_step(), adapted for documentation
    """

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.default_step_size = 10000

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.result.fill(-np.inf)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            indices = ctx.table.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            input_df = ctx.table.data()
            op = input_df.loc[fix_loc(indices)].max(keepdims=False)
            if self.result is None:
                self.result = PDict(op)
            else:
                for k, v in self.result.items():
                    self.result[k] = np.maximum(op[k], v)
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)


class TestMinMax(ProgressiveTest):
    def te_st_min(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10000, scheduler=s)
        min_ = Min(name="min_" + str(hash(random)), scheduler=s)
        min_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        assert random.result is not None
        assert min_.result is not None
        res1 = random.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def compare(self, res1: Dict[str, Any], res2: Dict[str, Any]) -> None:
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        self.assertTrue(np.allclose(v1, v2))

    def test_max(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10000, scheduler=s)
        max_ = Max(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert random.result is not None
        assert max_.result is not None
        res1 = random.result.max()
        res2 = max_.result
        self.compare(res1, res2)


if __name__ == "__main__":
    ProgressiveTest.main()
