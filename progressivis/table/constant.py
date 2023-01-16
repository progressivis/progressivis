from __future__ import annotations

from progressivis.utils.errors import ProgressiveStopIteration
from progressivis.core.module import ReturnRunStep
from .table import PTable
from .module import PTableModule
from ..utils.psdict import PDict

from typing import Union, Any


class Constant(PTableModule):
    def __init__(self, table: Union[None, PTable, PDict], **kwds: Any) -> None:
        super(Constant, self).__init__(**kwds)
        assert table is None or isinstance(table, (PTable, PDict))
        self.result = table

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        raise ProgressiveStopIteration()
