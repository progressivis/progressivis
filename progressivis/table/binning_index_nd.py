"""Binning Index computes an index for numerical values by
maintaining multiple pintsets for value ranges, each PIntSet corresponds
to a value bin. It differs from HistogramIndex because bins have all the same width
"""
from __future__ import annotations

import operator
import logging

import numpy as np
from ..core.pintset import PIntSet
from ..core.utils import slice_to_arange, fix_loc
from .. import ProgressiveError
from ..utils.errors import ProgressiveStopIteration
from ..utils.psdict import PDict
from ..core.module import (
    Module,
    ReturnRunStep,
    def_input,
    def_output,
    def_parameter,
    document,
)
from ..table.api import PTable, PTableSelectedView
from ..core.utils import indices_len
from .binning_index import _BinningIndexImpl

from typing import Optional, Any, Callable, Sequence, Generator

APPROX = False
logger = logging.getLogger(__name__)


@document
@def_parameter(
    "tol",
    np.dtype(float),
    -1,
    doc=(
        "Tolerance determining the bins width."
        "Negative values represent %, positive values are absolute"
    ),
)
@def_parameter("init_threshold", np.dtype(int), 10_000)
@def_parameter("max_trials", np.dtype(int), 5)
@def_input("table", PTable, hint_type=Sequence[str])
@def_output("result", PTableSelectedView)
@def_output("bin_timestamps_0", PDict, required=False)
@def_output("bin_timestamps_1", PDict, required=False)
@def_output("min_out", PDict, required=False)
@def_output("max_out", PDict, required=False)
class BinningIndexND(Module):
    """
    Compute and maintain a binned index by dividing the entire range of
    values of a column into a series of intervals (binning).
    Each bin contains the set of indices of rows in the underlying interval
    in order to  provide fast access to these rows.
    Actually `BinningIndexND` is able to build multiple indices, one for each
    column provided in the **table** slot hint.
    """

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        # will be created when the init_threshold is reached
        self._impl: dict[str, _BinningIndexImpl] | None = None
        self.selection = PIntSet()  # will be filled when the table is read
        # so realistic initial values for min and max were available
        self.input_module: Optional[Module] = None
        self.input_slot: Optional[str] = None
        self._input_table: Optional[PTable] = None
        self._prev_len = 0
        self._trials = 0
        self._columns: Sequence[str] | None = None

    def get_min_bin(self) -> dict[str, PIntSet | None] | None:
        if self._impl is None or self._columns is None:
            return None
        return {col: self._impl[col].get_min_bin() for col in self._columns}

    def get_max_bin(self) -> dict[str, PIntSet | None] | None:
        if self._impl is None or self._columns is None:
            return None
        return {col: self._impl[col].get_max_bin() for col in self._columns}

    def starting(self) -> None:
        super().starting()
        if self.get_output_slot("bin_timestamps_0"):
            self.bin_timestamps_0 = PDict()
        if self.get_output_slot("bin_timestamps_1"):
            self.bin_timestamps_1 = PDict()
        if self.get_output_slot("min_out"):
            self.min_out = PDict()
        if self.get_output_slot("max_out"):
            self.max_out = PDict()

    def feed_bin_timestamps(self) -> None:
        if self.bin_timestamps_0 is None:
            return
        if self.bin_timestamps_1 is None:
            return
        if self._columns is None:
            return
        assert len(self._columns)  # at least 1 col
        impl = self._impl
        assert impl is not None
        col = self._columns[0]
        for hit in impl[col].binvect_hits:  # PDict hasn't a defaultdict subclass so ...
            self.bin_timestamps_0[hit] = self.bin_timestamps_0.get(hit, 0) + 1  # type: ignore
        self.bin_timestamps_0[-1] = impl[col].origin  # type: ignore
        if len(self._columns) == 1:
            return
        col = self._columns[1]
        for hit in impl[col].binvect_hits:  # PDict hasn't a defaultdict subclass so ...
            self.bin_timestamps_1[hit] = self.bin_timestamps_1.get(hit, 0) + 1  # type: ignore
        self.bin_timestamps_1[-1] = impl[col].origin  # type: ignore

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        assert input_slot is not None
        steps = 0
        input_table = input_slot.data()
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        self._input_table = input_table
        len_table = len(input_table)
        if (
            len_table < self.params.init_threshold
            and not input_slot.output_module.is_terminated()
        ):
            # there are not enough rows. it's not worth building an index yet
            if self._trials > self.params.max_trials:
                print("init_threshold to high in BinningIndexND")
                raise ProgressiveStopIteration(
                    "init_threshold too high in BinningIndexND"
                )
            if len_table == self._prev_len:
                self._trials += 1
            self._prev_len = len_table
            return self._return_run_step(self.state_blocked, steps_run=0)
        if not self._columns:
            if (hint := input_slot.hint) is not None:
                assert len(hint)
                self._columns = hint
            else:
                raise ProgressiveError("BinningIndexND needs at least one column")
        if self._impl is None:
            input_slot.reset()
            input_slot.update(run_number)
            input_slot.clear_buffers()
            assert self._columns
            self._impl = {
                col: _BinningIndexImpl(col, input_table, self.params.tol)
                for col in self._columns
            }
            steps += self.process_min_max(input_table)
            self.selection = PIntSet(input_table.index)
            self.result = PTableSelectedView(input_table, self.selection)
            self.feed_bin_timestamps()
            return self._return_run_step(self.state_blocked, len(self.selection))
        impl = self._impl
        deleted: Optional[PIntSet] = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(as_slice=False)
            steps = 1
            deleted = fix_loc(deleted)
            self.selection -= deleted
        created: Optional[PIntSet] = None
        if input_slot.created.any():
            created = input_slot.created.next(length=step_size, as_slice=False)
            created = fix_loc(created)
            steps += indices_len(created)
            self.selection |= created
        updated: Optional[PIntSet] = None
        if input_slot.updated.any():
            updated = input_slot.updated.next(length=step_size, as_slice=False)
            updated = fix_loc(updated)
            steps += indices_len(updated)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        input_table = input_slot.data()
        assert self._columns
        for col in self._columns:
            impl[col].update_histogram(created, updated, deleted)
        self.feed_bin_timestamps()
        steps += self.process_min_max(input_table)
        return self._return_run_step(self.next_state(input_slot), steps_run=steps)

    def process_min_max(self, input_table: PTable) -> int:
        steps = 0
        assert self._columns is not None
        if self.min_out is not None:
            min_bin_nd = self.get_min_bin()
            if min_bin_nd:
                for col, min_bin in min_bin_nd.items():
                    min_ = input_table[col].loc[min_bin].min()
                    self.min_out.update({col: min_})
                assert min_bin
                steps += len(min_bin)  # TODO: find a better heuristic
        if self.max_out is not None:
            max_bin_nd = self.get_max_bin()
            if max_bin_nd:
                for col, max_bin in max_bin_nd.items():
                    max_ = input_table[col].loc[max_bin].max()
                    self.max_out.update({col: max_})
                assert max_bin
                steps += len(max_bin)  # TODO: find a better heuristic
        return steps

    def _eval_to_ids(
        self,
        column: str,
        operator_: Callable[[Any, Any], Any],
        limit: Any,
        input_ids: Optional[slice] = None,
    ) -> PIntSet:
        input_slot = self.get_input_slot("table")
        table_ = input_slot.data()
        if input_ids is None:
            input_ids = table_.index
        else:
            input_ids = fix_loc(input_ids)
        x = table_[column].loc[input_ids]
        mask_ = operator_(x, limit)
        assert isinstance(input_ids, slice)
        arr = slice_to_arange(input_ids)
        return PIntSet(arr[np.nonzero(mask_)[0]])

    def impl_col(self, column: str) -> _BinningIndexImpl | None:
        if self._impl is None or column not in self._impl:
            return None
        return self._impl[column]

    def query(
        self,
        column: str,
        operator_: Callable[[Any, Any], Any],
        limit: Any,
        approximate: bool = APPROX,
        only_bins: PIntSet = PIntSet(),
    ) -> PIntSet:
        """
        Return the list of rows matching the query.
        For example, returning all values less than 10 (< 10) would be
        `query(operator.__lt__, 10)`
        """
        if implc := self.impl_col(column):
            return implc.query(operator_, limit, approximate, only_bins)
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return self._eval_to_ids(column, operator_, limit)

    def restricted_query(
        self,
        column: str,
        operator_: Callable[[Any, Any], Any],
        limit: Any,
        only_locs: Any,
        approximate: bool = APPROX,
        only_bins: PIntSet = PIntSet(),
    ) -> PIntSet:
        """
        Return the list of rows matching the query.
        For example, returning all values less than 10 (< 10) would be
        `query(operator.__lt__, 10)`
        """
        if implc := self.impl_col(column):
            return implc.restricted_query(
                operator_, limit, only_locs, approximate, only_bins
            )
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return self._eval_to_ids(column, operator_, limit, only_locs)

    def range_query(
        self,
        column: str,
        lower: float,
        upper: float,
        approximate: bool = APPROX,
    ) -> PIntSet:
        r"""
        Return the list of rows with values in range \[`lower`, `upper`\[
        """
        if implc := self.impl_col(column):
            return implc.range_query(lower, upper, approximate)
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return self._eval_to_ids(
            column,
            operator.__lt__, upper
        ) & self._eval_to_ids(  # optimize later
            column,
            operator.__ge__, lower
        )

    def range_query_asgen(
        self,
        column: str,
        lower: float,
        upper: float,
        approximate: bool = APPROX,
    ) -> Generator[PIntSet, None, None]:
        r"""
        Return the list of rows with values in range \[`lower`, `upper`\[
        """
        if implc := self.impl_col(column):
            return implc.range_query_asgen(lower, upper, approximate)

        def never() -> Generator[PIntSet, None, None]:
            if False:
                yield PIntSet()

        return never()

    def restricted_range_query(
        self,
        column: str,
        lower: float,
        upper: float,
        only_locs: Any,
        approximate: bool = APPROX,
        only_bins: PIntSet = PIntSet(),
    ) -> PIntSet:
        r"""
        Return the list of rows with values in range \[`lower`, `upper`\[
        among only_locs
        """
        if implc := self.impl_col(column):
            return implc.restricted_range_query(
                lower, upper, only_locs, approximate, only_bins
            )
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return (
            self._eval_to_ids(column, operator.__lt__, upper, only_locs)
            # optimize later
            & self._eval_to_ids(column, operator.__ge__, lower, only_locs)
        )

    def compute_percentiles(
        self, column: str, points: dict[str, float], accuracy: float
    ) -> dict[str, float]:
        implc = self.impl_col(column)
        assert implc
        return implc.compute_percentiles(points, len(self.selection), accuracy)
        return {}

    def create_dependent_modules(
        self, input_module: Module, input_slot: str, **kwds: Any
    ) -> BinningIndexND:
        with self.grouped():
            self.input_module = input_module
            self.input_slot = input_slot
            hist_index = self
            hist_index.input.table = input_module.output[input_slot]
            return hist_index
