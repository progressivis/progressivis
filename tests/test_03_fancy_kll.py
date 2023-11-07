from . import ProgressiveTest, skipIf
import os
from progressivis.core import aio, Sink
from progressivis.utils.psdict import PDict
from progressivis.core.module import Module, def_input, def_output, ReturnRunStep
from progressivis.stats.fancy_kll import FancyKLL
from progressivis.stats import RandomPTable
import numpy as np
from typing import Any, Dict, List, Sequence, Union

K = 300
BINS = 128
QUANTILES = [0.3, 0.5, 0.7]
SPLITS_SEQ = [0.3, 0.5, 0.7]
SPLITS_DICT = dict(lower=0.1, upper=0.9, n_splits=10)

ArrayLike = Union[np.ndarray[Any, Any], Sequence[Any]]


@def_input("table", PDict, hint_type=Dict[str, List[float]])
@def_output("result", PDict)
class Double(Module):
    """
    Computes the double of all input values
    """

    def __init__(
        self,
        **kwds: Any,
    ) -> None:
        """
        Args:
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)
        self.default_step_size = 10000

    def is_ready(self) -> bool:
        slot = self.get_input_slot("table")
        if slot is not None and slot.created.any():
            return True
        return super().is_ready()

    def reset(self) -> None:
        if self.result is not None:
            self.result.fill(-np.inf)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        slot = self.get_input_slot("table")
        assert slot is not None
        if slot.has_buffered():
            slot.clear_buffers()
        data = slot.data()
        if data is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        dbl = {k: 2 * v for (k, v) in data.items()}
        if self.result is None:
            self.result = PDict(dbl)
        else:
            self.result.update(dbl)
        return self._return_run_step(self.next_state(slot), steps_run=len(dbl))


@skipIf(os.getenv("CI"), "randomly fails on CI")
class TestFancyKll(ProgressiveTest):
    def test_fancy_kll(self) -> None:
        np.random.seed(42)
        s = self.scheduler()
        random = RandomPTable(10, rows=10_000, scheduler=s)
        kll = FancyKLL(scheduler=s)
        kll.input.table = random.output.result
        dbl1 = Double(scheduler=s)
        dbl1.input.table = kll.output.result[{"_1": [0.1, 0.2], "_2": [0.3, 0.4]}]
        dbl2 = Double(scheduler=s)
        dbl2.input.table = kll.output.result[{"_2": [0.3, 0.4], "_3": [0.7, 0.8]}]
        sink = Sink(scheduler=s)
        sink.input.inp = dbl1.output.result
        sink.input.inp = dbl2.output.result
        aio.run(s.start())

    def compare(self, res1: ArrayLike, res2: ArrayLike, atol: float = 1e-02) -> None:
        v1 = np.array(res1)
        v2 = np.array(res2)
        self.assertEqual(v1.shape, v2.shape)
        self.assertTrue(np.allclose(v1, v2, atol=atol))


if __name__ == "__main__":
    ProgressiveTest.main()
