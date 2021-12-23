from __future__ import annotations

from progressivis.utils.errors import ProgressiveStopIteration
from .table import Table
from .module import TableModule, ReturnRunStep
from ..utils.psdict import PsDict

from typing import Union


class Constant(TableModule):
    def __init__(self,
                 table: Union[None, Table, PsDict], **kwds):
        super(Constant, self).__init__(**kwds)
        assert table is None or isinstance(table, (Table, PsDict))
        self.result = table

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        raise ProgressiveStopIteration()
