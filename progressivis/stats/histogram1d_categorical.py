from __future__ import annotations

import logging

import pandas as pd

from ..core.module import ReturnRunStep, def_input, def_output, document, Module
from ..core.utils import indices_len, fix_loc
from ..core.quality import QualitySqrtSumSquarredDiffs
from ..table.table import PTable
from ..utils.psdict import PDict
from ..core.decorators import process_slot, run_if_any
from ..utils.errors import ProgressiveError

from typing import Any, Union, Dict, cast


logger = logging.getLogger(__name__)


@document
@def_input("table", PTable, doc="the input table")
@def_output(
    "result",
    PDict,
    doc="occurence counters dictionary where every key represents a categorical value",
)
class Histogram1DCategorical(Module):
    """
    Compute the histogram (i.e. the bar chart) of a categorical column in the input table
    """

    def __init__(self, column: Union[str, int], **kwds: Any) -> None:
        """
        Args:
            column: the name or the position of the column to be processed
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(dataframe_slot="table", **kwds)
        self.column = column
        self.total_read = 0
        self.default_step_size = 1000
        self.result = PDict()
        self._quality: QualitySqrtSumSquarredDiffs | None = None

    def reset(self) -> None:
        if self.result:
            self.result.clear()

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        assert self.context is not None
        with self.context as ctx:
            dfslot = ctx.table
            if not dfslot.created.any():
                logger.info("Input buffers empty")
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            column = input_df[self.column]
            indices = dfslot.created.next(step_size)  # returns a slice or ...
            steps = indices_len(indices)
            logger.info("Read %d rows", steps)
            self.total_read += steps
            column = column.loc[fix_loc(indices)]
            if column.dtype != "O":
                raise ProgressiveError(f"Type of '{self.column}' is" " not string")
            valcounts = pd.Series(column).value_counts().to_dict()
            assert self.result is not None
            for k, v in valcounts.items():
                key = cast(str, k)
                self.result[key] = v + self.result.get(key, 0)
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)

    def is_visualization(self) -> bool:
        return True

    def get_visualization(self) -> str:
        return "histogram1d_categorical"

    def get_quality(self) -> Dict[str, float] | None:
        assert self.result is not None
        if len(self.result) == 0:
            return None
        array = self.result.array
        if self._quality is None:
            self._quality = QualitySqrtSumSquarredDiffs()
        return {
            "histogram1d_cat_"
            + str(self.column): self._quality.quality(array / array.sum())  # normalize
        }
