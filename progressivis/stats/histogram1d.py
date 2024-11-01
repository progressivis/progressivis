from __future__ import annotations

from ..core.module import (
    JSon,
    ReturnRunStep,
    def_input,
    def_output,
    def_parameter,
    document,
)
from ..core.utils import indices_len, fix_loc, integer_types
from ..core.slot import Slot
from ..core.module import Module
from ..table.table import PTable
from ..utils.psdict import PDict
from ..core.decorators import process_slot, run_if_any
import numpy as np

import logging

from typing import Any, Sequence


logger = logging.getLogger(__name__)


@document
@def_parameter(
    "bins",
    np.dtype(int),
    128,
    doc=(
        "the number of equal-width bins in the range given by the min/max inputs"
        "interval"
    ),
)
@def_parameter(
    "delta",
    np.dtype(float),
    -5,
    doc=(
        "tolerance threshold for variations in the min/max values at which the bounds "
        "are changed. Negative values represent %, positive values are absolute"
    ),
)
@def_input("table", PTable, hint_type=Sequence[str], doc="the input table")
@def_input(
    "min",
    PDict,
    doc="The minimum value in the input data. It could be provided by a {{Min}} module",
)
@def_input(
    "max",
    PDict,
    doc="The maximum value in the input data. It could be provided by a {{Max}} module",
)
@def_output(
    "result",
    PDict,
    datashape={"array": np.ndarray, "min": float, "max": float, "time": int},
    doc=(
        "the output table. Its datashape is"
        " ``{ array: var * int32, min: float64, max: float64, time: int64 }``"
    ),
)
class Histogram1D(Module):
    """
    Compute the histogram of a scalar, numerical column in the input table
    """

    schema = "{ array: var * int32, min: float64, max: float64, time: int64 }"

    def __init__(self, column: int | str | None = None, **kwds: Any) -> None:
        """
        Args:
            column: the name or the position of the column to be processed
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """

        super().__init__(dataframe_slot="table", **kwds)
        self.tags.add(self.TAG_VISUALIZATION)
        self.column = column
        self.total_read = 0
        self.default_step_size = 1000
        self._histo: np.ndarray[Any, Any] | None = None
        self._edges: np.ndarray[Any, Any] | None = None
        self._bounds: tuple[float, float] | None = None
        self._h_cnt = 0
        self.result = PDict()

    def reset(self) -> None:
        self._histo = None
        self._edges = None
        self._bounds = None
        self.total_read = 0
        self._h_cnt = 0
        self.get_input_slot("table").reset()
        if self.result:
            self.result.clear()

    def is_ready(self) -> bool:
        if self._bounds and self.get_input_slot("table").created.any():
            return True
        return super().is_ready()

    @process_slot("table", reset_cb="reset")
    @process_slot("min", "max", reset_if=False)
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        assert self.result is not None
        with self.context as ctx:
            dfslot = ctx.table
            min_slot = ctx.min
            max_slot = ctx.max
            if not (
                dfslot.created.any()
                or min_slot.has_buffered()
                or max_slot.has_buffered()
            ):
                logger.info("Input buffers empty")
                return self._return_run_step(self.state_blocked, steps_run=0)
            if self.column is None:
                assert dfslot.hint is not None
                assert len(dfslot.hint) == 1
                self.column = dfslot.hint[0]
            min_slot.clear_buffers()
            max_slot.clear_buffers()
            bounds = self.get_bounds(min_slot, max_slot)
            if bounds is None:
                logger.debug("No bounds yet at run %d", run_number)
                return self._return_run_step(self.state_blocked, steps_run=0)
            bound_min, bound_max = bounds
            if self._bounds is None:
                delta = self.get_delta(*bounds)
                self._bounds = (bound_min - delta, bound_max + delta)
                logger.info("New bounds at run %d: %s", run_number, self._bounds)
            else:
                (old_min, old_max) = self._bounds
                delta = self.get_delta(*bounds)
                if (
                    bound_min < old_min
                    or bound_max > old_max
                    or bound_min > (old_min + delta)
                    or bound_max < (old_max - delta)
                ):
                    self._bounds = (bound_min - delta, bound_max + delta)
                    logger.info(
                        "Updated bounds at run %d: %s", run_number, self._bounds
                    )
                    dfslot.reset()
                    dfslot.update(run_number)
                    min_slot.reset()
                    min_slot.update(run_number)
                    max_slot.reset()
                    max_slot.update(run_number)
                    self.reset()
                    return self._return_run_step(self.state_blocked, steps_run=0)
            (curr_min, curr_max) = self._bounds
            if curr_min >= curr_max:
                logger.error("Invalid bounds: %s", self._bounds)
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            indices = dfslot.created.next(length=step_size)  # returns a slice or ...
            steps = indices_len(indices)
            logger.info("Read %d rows", steps)
            self.total_read += steps
            column = input_df[self.column]
            column = column.loc[fix_loc(indices)]
            bins = self.params.bins
            if self._edges is not None and len(self._edges) != bins + 1:
                self.reset()
                return self._return_run_step(self.state_blocked, steps_run=0)
            histo = None
            if len(column) > 0:
                histo, self._edges = np.histogram(
                    column, bins=bins, range=(curr_min, curr_max), density=False
                )
                self._h_cnt += len(column)
            if self._histo is None:
                self._histo = histo
            elif histo is not None:
                self._histo += histo
            values = {
                "array": self._histo,
                "min": curr_min,
                "max": curr_max,
                "time": run_number,
            }
            self.result.update(values)
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)

    def get_bounds(
        self, min_slot: Slot, max_slot: Slot
    ) -> tuple[float, float] | None:
        min_df = min_slot.data()
        if len(min_df) == 0 and self._bounds is None:
            return None
        min_ = min_df[self.column]
        max_df = max_slot.data()
        if len(max_df) == 0 and self._bounds is None:
            return None
        max_ = max_df[self.column]
        return (min_, max_)

    def get_delta(self, min_: float, max_: float) -> float:
        delta: float = self.params["delta"]
        extent = max_ - min_
        if delta < 0:
            return extent * delta / -100.0
        return 0

    def get_histogram(self) -> dict[str, Any]:
        min_ = self._bounds[0] if self._bounds else None
        max_ = self._bounds[1] if self._bounds else None
        edges: Any = self._edges
        if edges is None:
            edges = []
        elif isinstance(edges, integer_types):
            edges = [edges]
        else:
            edges = edges.tolist()
        return {
            "edges": edges,
            "values": self._histo.tolist() if self._histo is not None else [],
            "min": min_,
            "max": max_,
        }

    def get_visualization(self) -> str:
        return "histogram1d"

    def to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
        json = super().to_json(short, with_speed)
        if short:
            return json
        return self._hist_to_json(json)

    def _hist_to_json(self, json: JSon) -> JSon:
        json["histogram"] = self.get_histogram()
        return json
