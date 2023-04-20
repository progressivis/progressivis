from __future__ import annotations

from progressivis.utils.errors import ProgressiveStopIteration
from progressivis.core.module import ReturnRunStep, def_output, document
from .table import PTable
from ..core.module import Module
from ..utils.psdict import PDict

from typing import Optional, Any


@document
@def_output("result", PTable)
class Constant(Module):
    def __init__(self, table: Optional[PTable], **kwds: Any) -> None:
        """
        Args:
            table:
                table to be used by the **result** output slot
        """
        super().__init__(**kwds)
        assert table is None or isinstance(table, PTable)
        self.result = table

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        raise ProgressiveStopIteration()


@def_output("result", PDict)
class ConstDict(Module):
    def __init__(self, pdict: Optional[PDict], **kwds: Any) -> None:
        super().__init__(**kwds)
        assert pdict is None or isinstance(pdict, PDict)
        self.result = pdict

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        raise ProgressiveStopIteration()
