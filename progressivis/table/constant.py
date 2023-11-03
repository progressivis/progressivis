from __future__ import annotations

from progressivis.utils.errors import ProgressiveStopIteration
from progressivis.core.module import ReturnRunStep, def_output, document
from .table import PTable
from ..core.module import Module
from ..utils.psdict import PDict

from typing import Any


@document
@def_output("result", PTable)
class Constant(Module):
    """
    Module providing a constant output {{PTable}} slot
    """
    def __init__(self, table: PTable, **kwds: Any) -> None:
        """
        Args:
            table: object to be used by the ``result`` output slot

        """
        super().__init__(**kwds)
        assert isinstance(table, PTable)
        self.result = table

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        raise ProgressiveStopIteration()


@document
@def_output("result", PDict)
class ConstDict(Module):
    """
    Module providing a constant output {{PDict}} slot
    """
    def __init__(self, pdict: PDict, **kwds: Any) -> None:
        """
        Args:
            pdict: object to be used by the **result** output slot
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)
        assert isinstance(pdict, PDict)
        self.result = pdict

    def predict_step_size(self, duration: float) -> int:
        return 1

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        raise ProgressiveStopIteration()
