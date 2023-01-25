from __future__ import annotations

from progressivis.utils.errors import ProgressiveStopIteration
from progressivis.core.module import ReturnRunStep
from .table import PTable
from .module import PTableModule, PDictModule
from ..utils.psdict import PDict

from typing import Union, Any


class Constant(PTableModule):
    def __init__(self, table: Union[None, PTable], **kwds: Any) -> None:
        super().__init__(**kwds)
        assert table is None or isinstance(table, PTable)
        self.result = table

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        raise ProgressiveStopIteration()


class ConstDict(PDictModule):
    def __init__(self, pdict: Union[None, PDict], **kwds: Any) -> None:
        super().__init__(**kwds)
        assert pdict is None or isinstance(pdict, PDict)
        self.result = pdict

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        raise ProgressiveStopIteration()
