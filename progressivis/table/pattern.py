from __future__ import annotations

from progressivis.utils.errors import ProgressiveStopIteration
from progressivis.core.module import Module, ReturnRunStep, def_output
from .table import PTable
from ..core import Sink
from typing import Any, Optional


@def_output("result", PTable)
class Pattern(Module):
    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self.sink: Optional[Sink] = None

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        raise ProgressiveStopIteration()

    def create_dependent_modules(self) -> None:
        s = self.scheduler()
        self.dep.sink = Sink(scheduler=s)
        self.dep.sink.input.inp = self.output.result
