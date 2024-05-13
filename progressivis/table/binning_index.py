"""Binning Index computes an index for numerical values by
maintaining multiple pintsets for value ranges, each PIntSet corresponds
to a value bin. It differs from HistogramIndex because bins have all the same width
"""
from __future__ import annotations

import operator
import logging

import numpy as np
from collections import deque
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
from . import PTable
from ..core.utils import indices_len
from . import PTableSelectedView


from typing import Optional, Any, Callable, Sequence, cast, Generator

APPROX = False
logger = logging.getLogger(__name__)

BinVect = deque


def _union(*args: Any) -> PIntSet:
    try:
        return PIntSet.union(*args)
    except TypeError:
        if not args:
            return PIntSet()
        raise


class _BinningIndexImpl:
    "Implementation part of Histogram Index"

    # pylint: disable=too-many-instance-attributes
    def __init__(self, column: str, table: PTable, tol: float):
        self.name = column
        self.column = table[column]
        self.binvect: BinVect[PIntSet | None] = BinVect([])
        self.binvect_map: PIntSet = PIntSet()
        self.binvect_hits: PIntSet = PIntSet()
        self._sampling_size = 1000
        self.tol = tol
        self.origin: float = 0.0
        self.bin_w: float = 0.0
        self._initialize()
        self.update_histogram(created=table.index)

    def _initialize(self) -> None:
        assert self.tol != 0
        min_ = self.column.min()
        max_ = self.column.max()
        self.origin = min_
        if self.tol > 0:
            self.bin_w = self.tol
        else:
            q5 = np.percentile(self.column, 5)  # type: ignore
            q95 = np.percentile(self.column, 95)  # type: ignore
            self.bin_w = (q95 - q5) * (abs(self.tol) / 100)
        assert self.bin_w > 0
        binvect_size = int((max_ - min_) / self.bin_w) + 1
        self.binvect = BinVect([None] * binvect_size)

    def update_histogram(
        self,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
    ) -> None:
        "Update the histogram index"
        created = PIntSet.aspintset(created)
        updated = PIntSet.aspintset(updated)
        deleted = PIntSet.aspintset(deleted)
        self.binvect_hits = PIntSet()
        if deleted or updated:
            to_remove = updated | deleted
            for i in self.binvect_map:
                if self.binvect[i] & to_remove:  # type: ignore
                    self.binvect[i] -= to_remove  # type: ignore
                    self.binvect_hits.add(i)
        if created or updated:
            to_add = created | updated
            ids = np.array(to_add, np.int64)
            values = self.column.loc[to_add]
            i_bins = np.array((values - self.origin) // self.bin_w, dtype=int)
            if np.any(i_bins < 0):
                print("Origin changed")
                min_i = i_bins.min()
                assert min_i < 0
                offset = -min_i
                self.binvect.extendleft([None] * offset)
                self.origin -= self.bin_w * offset
                i_bins += offset
                self.binvect_map = PIntSet([elt + offset for elt in self.binvect_map])
                self.binvect_hits = PIntSet([elt + offset for elt in self.binvect_hits])
            argsort_i = np.argsort(i_bins)
            uv, ui = np.unique(i_bins[argsort_i], return_index=True)
            split_ = np.split(ids[argsort_i], ui[1:])
            for bin_id, ids in zip(uv, split_):
                if not ids.shape[0]:
                    continue
                try:
                    self.binvect[bin_id] |= PIntSet(ids)  # type: ignore
                except TypeError:  # unsupported operand type(s) for |=: 'NoneType' and
                    if self.binvect[bin_id] is not None:
                        raise
                    self.binvect[bin_id] = PIntSet(ids)
                except IndexError:  # deque index out of range
                    if bin_id < len(self.binvect):
                        raise
                    self.binvect.extend([None] * (bin_id - len(self.binvect) + 1))
                    self.binvect[bin_id] = PIntSet(ids)
                self.binvect_map.add(bin_id)
                self.binvect_hits.add(bin_id)

    def query(
        self,
        operator_: Callable[[Any, Any], int],
        limit: Any,
        approximate: bool = APPROX,
        only_bins: PIntSet = PIntSet(),
    ) -> PIntSet:
        """
        Return the list of rows matching the query.
        For example, returning all values less than 10 (< 10) would be
        `query(operator.__lt__, 10)`
        """
        assert self.binvect is not None
        binvect, origin, bin_w = self.binvect, self.origin, self.bin_w
        pos = int((limit - origin) // bin_w)
        if only_bins:
            selected_bins = (
                cast(PIntSet, binvect[i])
                for i in self.binvect_map
                if i in only_bins and operator_(i, pos)
            )

        else:
            selected_bins = (
                cast(PIntSet, binvect[i])
                for i in self.binvect_map if operator_(i, pos)
            )

        union = _union(*selected_bins)
        detail = PIntSet()
        if not approximate:
            ids = np.array(self.binvect[pos], np.int64)
            values = self.column.loc[ids]
            selected = ids[operator_(values, limit)]
            detail.update(selected)
            union.update(detail)
        return union

    def restricted_query(
        self,
        operator_: Callable[[Any, Any], int],
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
        assert self.binvect is not None
        binvect, origin, bin_w = self.binvect, self.origin, self.bin_w
        pos = int((limit - origin) // bin_w)
        if only_bins:
            selected_bins = (
                cast(PIntSet, binvect[i])
                for i in self.binvect_map
                if i in only_bins and operator_(i, pos) and binvect[i] & only_locs
            )
        else:
            selected_bins = (
                cast(PIntSet, binvect[i])
                for i in self.binvect_map if operator_(i, pos) and binvect[i] & only_locs
            )

        union = _union(*selected_bins)
        detail = PIntSet()
        if not approximate:
            ids = np.array(self.binvect[pos], np.int64)
            values = self.column.loc[ids]
            selected = ids[operator_(values, limit)]
            detail.update(selected)
            union.update(detail)
        return union

    def range_query_(
        self,
        lower: float,
        upper: float,
        approximate: bool = APPROX,
        only_bins: PIntSet = PIntSet(),
    ) -> tuple[Generator[PIntSet, None, None], PIntSet]:
        """
        Return the PIntSet of all rows with values in range [`lower`, `upper`[
        """
        if lower > upper:
            lower, upper = upper, lower
        assert self.binvect is not None
        binvect, origin, bin_w = self.binvect, self.origin, self.bin_w
        lower_bin = int((lower - origin) // bin_w)
        pos_lo = lower_bin
        upper_bin = int((upper - origin) // bin_w)
        pos_up = upper_bin
        if not approximate:  # i.e. precise
            lower_bin += 1
        assert lower_bin <= upper_bin
        if only_bins:
            selected_bins = (
                cast(PIntSet, binvect[i])
                for i in self.binvect_map
                if i in only_bins and i >= lower_bin and i < upper_bin
            )
        else:
            selected_bins = (
                cast(PIntSet, binvect[i])
                for i in self.binvect_map if i >= lower_bin and i < upper_bin
            )

        # union = PIntSet.union(*selected_bins)  # type: ignore
        detail = PIntSet()
        if not approximate:

            ids = np.array(self.binvect[pos_lo], np.int64)
            values = self.column.loc[ids]
            if pos_lo == pos_up:
                selected = ids[(lower <= values) & (values < upper)]
                detail.update(selected)
            else:
                selected = ids[lower <= values]
                detail.update(selected)
                ids = np.array(self.binvect[pos_up], np.int64)
                values = self.column.loc[ids]
                selected = ids[values < upper]
                detail.update(selected)
            # union.update(detail)
        return selected_bins, detail

    def range_query(
        self,
        lower: float,
        upper: float,
        approximate: bool = APPROX,
        only_bins: PIntSet = PIntSet(),
    ) -> PIntSet:
        selected_bins, detail = self.range_query_(lower, upper, approximate, only_bins)
        return _union(detail, *selected_bins)

    def range_query_aslist(
        self,
        lower: float,
        upper: float,
        approximate: bool = APPROX,
        only_bins: PIntSet = PIntSet(),
    ) -> Generator[PIntSet, None, None]:
        selected_bins, detail = self.range_query_(lower, upper, approximate, only_bins)
        if not detail:
            return selected_bins

        def _gen() -> Generator[PIntSet, None, None]:
            for elt in selected_bins:
                yield elt
            yield detail
        return _gen()

    def restricted_range_query(
        self,
        lower: float,
        upper: float,
        only_locs: Any,
        approximate: bool = APPROX,
        only_bins: PIntSet = PIntSet(),
    ) -> PIntSet:
        """
        Return the PIntSet of only_locs rows in range [`lower`, `upper`[
        """
        if lower > upper:
            lower, upper = upper, lower
        only_locs = PIntSet.aspintset(only_locs)
        binvect, origin, bin_w = self.binvect, self.origin, self.bin_w
        try:
            lower_bin = int((lower - origin) // bin_w)
            pos_lo = lower_bin
        except ValueError:
            if np.isinf(lower):
                pos_lo = self.binvect_map[0]
                lower_bin = pos_lo
            else:
                raise
        try:
            upper_bin = int((upper - origin) // bin_w)
            pos_up = upper_bin
        except ValueError:
            if np.isinf(upper):
                pos_up = self.binvect_map[-1]
                upper_bin = pos_up + 1
            else:
                raise
        # print("ratio:", len(only_bins), "/", len(self.binvect_map), len(only_bins)/len(self.binvect_map))
        if not approximate:  # i.e. precise
            lower_bin += 1
        if only_bins:
            selected_bins = (
                binvect[i]
                for i in self.binvect_map
                if i in only_bins
                and i >= lower_bin
                and i < upper_bin
                and (binvect[i] & only_locs)
            )
        else:
            selected_bins = (
                binvect[i]
                for i in self.binvect_map
                if i >= lower_bin and i < upper_bin and (binvect[i] & only_locs)
            )
        union = cast(PIntSet, _union(*selected_bins) & only_locs)
        if not approximate:  # i.e. precise
            detail = PIntSet()
            ids = np.array(self.binvect[pos_lo] & only_locs, np.int64)
            values = self.column.loc[ids]
            if pos_lo == pos_up:
                selected = ids[(lower <= values) & (values < upper)]
                detail.update(selected)
            else:
                selected = ids[lower <= values]
                detail.update(selected)
                ids = np.array(self.binvect[pos_up] & only_locs, np.int64)
                values = self.column.loc[ids]
                selected = ids[values < upper]
                detail.update(selected)
            union.update(detail)
        return union

    def get_min_bin(self) -> Optional[PIntSet]:
        for i in self.binvect_map:
            if bm := self.binvect[i]:
                return bm
        return None

    def get_max_bin(self) -> Optional[PIntSet]:
        for i in reversed(self.binvect_map):
            if bm := self.binvect[i]:
                return bm
        return None

    def compute_percentiles(self, points: dict[str, float], len_: int, accuracy: float) -> dict[str, float]:
        k_points = [p * (len_ + 1) * 0.01 for p in points.values()]
        max_k = max(k_points)
        ret_values: list[float] = []
        k_accuracy = accuracy * len_ * 0.01
        acc_vect = np.empty(len_, dtype=np.int64)
        size_vect = np.empty(len_, dtype=np.int64)
        acc = 0
        for i in self.binvect_map:
            val = len(cast(PIntSet, self.binvect[i]))
            size_vect[i] = val
            acc += val
            acc_vect[i] = acc
            if acc > max_k:
                break  # just avoids unnecessary computes
        acc_vect = acc_vect[: i + 1]
        for k in k_points:
            i = (acc_vect >= k).nonzero()[0][0]
            reminder = int(acc_vect[i] - k)
            assert size_vect[i] > reminder >= 0
            if size_vect[i] < k_accuracy:  # one takes the 1st value as a random one
                ret_values.append(self.column[cast(PIntSet, self.binvect[i])[0]])
            else:
                values = self.column.loc[self.binvect[i]]
                _ = np.partition(values, reminder)
                ret_values.append(values[reminder])
        return dict(zip(points.keys(), ret_values))


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
@def_output("bin_timestamps", PDict, required=False)
@def_output("min_out", PDict, required=False)
@def_output("max_out", PDict, required=False)
class BinningIndex(Module):
    """
    Compute and maintain a binned index by dividing the entire range of
    values of a column into a series of intervals (binning).
    Each bin contains the set of indices of rows in the underlying interval
    in order to  provide fast access to these rows.
    Actually `BinningIndex` is able to build multiple indices, one for each
    column provided in the **table** slot hint.
    """

    def __init__(self, column: str | None = None, **kwds: Any) -> None:
        super().__init__(**kwds)
        self._column = column
        # will be created when the init_threshold is reached
        self._impl: _BinningIndexImpl | None = None
        self.selection = PIntSet()  # will be filled when the table is read
        # so realistic initial values for min and max were available
        self.input_module: Optional[Module] = None
        self.input_slot: Optional[str] = None
        self._input_table: Optional[PTable] = None
        self._prev_len = 0
        self._trials = 0

    def get_min_bin(self) -> Optional[PIntSet]:
        if self._impl is None:
            return None
        return self._impl.get_min_bin()

    def get_max_bin(self) -> Optional[PIntSet]:
        if self._impl is None:
            return None
        return self._impl.get_max_bin()

    def starting(self) -> None:
        super().starting()
        if self.get_output_slot("bin_timestamps"):
            self.bin_timestamps = PDict()
        if self.get_output_slot("min_out"):
            self.min_out = PDict()
        if self.get_output_slot("max_out"):
            self.max_out = PDict()

    def feed_bin_timestamps(self) -> None:
        if self.bin_timestamps is None:
            return
        impl = self._impl
        assert impl is not None
        for hit in impl.binvect_hits:  # PDict hasn't a defaultdict subclass so ...
            self.bin_timestamps[hit] = self.bin_timestamps.get(hit, 0) + 1  # type: ignore
        self.bin_timestamps[-1] = impl.origin  # type: ignore

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        assert input_slot is not None
        steps = 0
        input_table = input_slot.data()
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        self._input_table = input_table
        len_table = len(input_table)
        if len_table < self.params.init_threshold:
            # there are not enough rows. it's not worth building an index yet
            if self._trials > self.params.max_trials:
                print("init_threshold to high in BinningIndex")
                raise ProgressiveStopIteration(
                    "init_threshold too high in BinningIndex"
                )
            if len_table == self._prev_len:
                self._trials += 1
            self._prev_len = len_table
            return self._return_run_step(self.state_blocked, steps_run=0)
        if not self._column:
            if (hint := input_slot.hint) is not None:
                assert len(hint) == 1
                self._column = hint[0]
            else:
                raise ProgressiveError("BinningIndex needs at least one column")
        if not self._impl:
            input_slot.reset()
            input_slot.update(run_number)
            input_slot.clear_buffers()
            assert self._column is not None
            self._impl = _BinningIndexImpl(
                self._column, input_table, self.params.tol
            )
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
        impl.update_histogram(created, updated, deleted)
        self.feed_bin_timestamps()
        steps += self.process_min_max(input_table)
        return self._return_run_step(self.next_state(input_slot), steps_run=steps)

    def process_min_max(self, input_table: PTable) -> int:
        steps = 0
        assert self._column is not None
        if self.min_out is not None:
            min_bin = self.get_min_bin()
            if min_bin:
                min_ = input_table[self._column].loc[min_bin].min()
                self.min_out.update({self._column: min_})
                steps += len(min_bin)  # TODO: find a better heuristic
        if self.max_out is not None:
            max_bin = self.get_max_bin()
            if max_bin:
                max_ = input_table[self._column].loc[max_bin].max()
                self.max_out.update({self._column: max_})
                steps += len(max_bin)  # TODO: find a better heuristic
        return steps

    def _eval_to_ids(
        self,
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
        x = table_[self._column].loc[input_ids]
        mask_ = operator_(x, limit)
        assert isinstance(input_ids, slice)
        arr = slice_to_arange(input_ids)
        return PIntSet(arr[np.nonzero(mask_)[0]])

    def query(
            self,
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
        if self._impl:
            return self._impl.query(operator_, limit, approximate, only_bins)
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return self._eval_to_ids(operator_, limit)

    def restricted_query(
            self,
            operator_: Callable[[Any, Any], Any],
            limit: Any,
            only_locs: Any,
            approximate: bool = APPROX,
            only_bins: PIntSet = PIntSet()
    ) -> PIntSet:
        """
        Return the list of rows matching the query.
        For example, returning all values less than 10 (< 10) would be
        `query(operator.__lt__, 10)`
        """
        if self._impl:
            return self._impl.restricted_query(operator_, limit, only_locs, approximate, only_bins)
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return self._eval_to_ids(operator_, limit, only_locs)

    def range_query(
            self, lower: float,
            upper: float,
            approximate: bool = APPROX,
            only_bins: PIntSet = PIntSet()
    ) -> PIntSet:
        r"""
        Return the list of rows with values in range \[`lower`, `upper`\[
        """
        if self._impl:
            return self._impl.range_query(
                lower, upper, approximate, only_bins
            )
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return self._eval_to_ids(
            operator.__lt__, upper
        ) & self._eval_to_ids(  # optimize later
            operator.__ge__, lower
        )

    def range_query_aslist(
            self, lower: float,
            upper: float,
            approximate: bool = APPROX,
            only_bins: PIntSet = PIntSet()
    ) -> Generator[PIntSet, None, None]:
        r"""
        Return the list of rows with values in range \[`lower`, `upper`\[
        """
        if self._impl:
            return self._impl.range_query_aslist(lower, upper, approximate, only_bins)

        def never() -> Generator[PIntSet, None, None]:
            if False:
                yield PIntSet()
        return never()

    def restricted_range_query(
        self,
        lower: float,
        upper: float,
        only_locs: Any,
        approximate: bool = APPROX,
        only_bins: PIntSet = PIntSet()
    ) -> PIntSet:
        r"""
        Return the list of rows with values in range \[`lower`, `upper`\[
        among only_locs
        """
        if self._impl:
            return self._impl.restricted_range_query(
                lower, upper, only_locs, approximate, only_bins
            )
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return (
            self._eval_to_ids(operator.__lt__, upper, only_locs)
            # optimize later
            & self._eval_to_ids(operator.__ge__, lower, only_locs)
        )

    def compute_percentiles(self, points: dict[str, float], accuracy: float) -> dict[str, float]:
        assert self._impl
        return self._impl.compute_percentiles(points, len(self.selection), accuracy)
        return {}

    def create_dependent_modules(
        self, input_module: Module, input_slot: str, **kwds: Any
    ) -> BinningIndex:
        with self.grouped():
            self.input_module = input_module
            self.input_slot = input_slot
            hist_index = self
            hist_index.input.table = input_module.output[input_slot]
            # hist_index.input.min = min_.output.result
            # hist_index.input.max = max_.output.result
            return hist_index
