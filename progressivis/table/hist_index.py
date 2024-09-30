"""Histogram Index computes a index for numerical values by
maintaining multiple pintsets for value ranges, each PIntSet corresponds
to a value bin.  The first bin corresponds to half infinite values
lower than the first specifid value, and the last bin corresponds to
half infinite values higher than the last specified value.
"""
from __future__ import annotations

import operator
import logging

import numpy as np
from ..core.pintset import PIntSet
from ..core.utils import slice_to_arange, fix_loc
from .. import ProgressiveError
from ..utils.psdict import PDict
from ..core.module import (
    Module,
    ReturnRunStep,
    def_input,
    def_output,
    def_parameter,
    document
)
from .api import PTable, PTableSelectedView
from ..core.utils import indices_len


from typing import List, Dict, Optional, Any, Callable, Tuple, Sequence

APPROX = False
logger = logging.getLogger(__name__)


class _HistogramIndexImpl(object):
    "Implementation part of Histogram Index"
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self, column: str, table: PTable, e_min: float, e_max: float, nb_bin: int
    ):
        self.column = table[column]
        self.e_min = e_min
        self.e_max = e_max
        self.pintsets: List[PIntSet] = []
        self.bins: Optional[np.ndarray[Any, Any]] = None
        self._sampling_size = 1000
        self._perm_deviation = 0.1  # permitted deviation
        self._divide_threshold = 1000  # TODO: make it settable!
        self._divide_coef = 5  # TODO: make it settable!
        self._merge_threshold = 1000  # TODO: make it settable!
        self._merge_coef = 5  # TODO: make it settable!
        self._min_hist_size = 64
        self._max_hist_size = 256
        self._init_histogram(e_min, e_max, nb_bin)
        self.update_histogram(created=table.index)

    def _init_histogram(self, e_min: float, e_max: float, nb_bin: int) -> None:
        self.bins = np.linspace(e_min, e_max, nb_bin, endpoint=True)
        assert len(self.bins) == nb_bin
        self.pintsets = [PIntSet() for _ in range(nb_bin + 1)]

    def _needs_division(self, size: int) -> bool:
        len_c = len(self.column)
        assert self.bins is not None
        len_b = len(self.bins)
        mean = float(len_c) / len_b
        if size < self._divide_threshold:
            return False
        return size > self._divide_coef * mean

    def show_histogram(self) -> None:
        "Print the histogram on the display"
        for i, bm in enumerate(self.pintsets):
            print(i, len(bm), "=" * len(bm))

    def find_bin(self, elt: Any) -> List[int]:
        res: List[int] = []
        for i, bm in enumerate(self.pintsets):
            if elt in bm:
                res.append(i)
        return res  # if len(res)>1: BUG

    def _is_merging_required(self) -> bool:
        return len(self.pintsets) > self._max_hist_size

    def _is_mergeable_pair(self, bm1: PIntSet, bm2: PIntSet, merge_cnt: int) -> bool:
        assert self.bins is not None
        if len(self.pintsets) - merge_cnt < self._min_hist_size:
            return False
        len_c = len(self.column)
        len_b = len(self.bins)
        mean = float(len_c) / len_b
        return len(bm1) + len(bm2) < max(self._merge_coef * mean, self._merge_threshold)

    def merge_once(self) -> int:
        assert self.bins is not None
        assert len(self.bins) + 1 == len(self.pintsets), "unexpected # of bins"
        bins_1 = list(self.bins)
        bins_1.append(None)
        bin_tuples = list(zip(self.pintsets, bins_1))
        merge_cnt = 0
        if len(bin_tuples) <= 2:
            return merge_cnt
        merged_tuples = []
        prev_bm, prev_sep = bin_tuples[0]
        for bm, sep in bin_tuples[1:]:
            if self._is_mergeable_pair(prev_bm, bm, merge_cnt):
                prev_bm = prev_bm | bm
                prev_sep = sep
                merge_cnt += 1
            else:
                merged_tuples.append((prev_bm, prev_sep))
                prev_bm, prev_sep = bm, sep
        assert prev_sep is None
        merged_pintsets: Sequence[Any]
        merged_pintsets, merged_bins = zip(*merged_tuples)
        merged_pintsets = list(merged_pintsets) + [prev_bm]
        self.bins = np.array(merged_bins)
        self.pintsets = merged_pintsets
        return merge_cnt

    def reshape(self) -> None:
        for i, bm in enumerate(self.pintsets):
            if self._needs_division(len(bm)):
                self.divide_bin(i)
        if self._is_merging_required():
            self.merge_once()

    def divide_bin(self, i: int) -> None:
        "Change the bounds of the index if needed"
        assert self.bins is not None

        ids = np.array(self.pintsets[i], np.int64)
        if self._sampling_size * 1.2 < len(ids):
            samples = np.random.choice(ids, self._sampling_size, replace=False)
        else:
            samples = ids
        s_vals = self.column.loc[samples]

        v = np.median(s_vals)
        if v == self.bins[i - 1] or (
            i < len(self.bins) and v == self.bins[i]
        ):  # there are a lot of identical
            return  # values=> do not divide
        if i >= len(self.bins):
            assert self.bins[i - 1] < v
        else:
            assert self.bins[i - 1] < v < self.bins[i] if i > 0 else v < self.bins[i]
        values = self.column.loc[ids]
        lower_bin = PIntSet(ids[values < v])
        upper_bin = self.pintsets[i] - lower_bin
        self.bins = np.insert(self.bins, i, v)
        assert self.bins is not None
        if i + 1 >= len(self.bins):
            assert self.bins[i - 1] < self.bins[i]
        else:
            assert (
                self.bins[i - 1] < self.bins[i] < self.bins[i + 1]
                if i > 0
                else self.bins[i] < self.bins[i + 1]
            )
        self.pintsets.insert(i, lower_bin)
        self.pintsets[i + 1] = upper_bin
        # print("*", end="")

    def _get_bin(self, val: float) -> PIntSet:
        i = np.digitize(val, self.bins)  # type: ignore
        return self.pintsets[int(i)]

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
        # if deleted:
        #     self._tdigest_is_valid = False
        if deleted or updated:
            to_remove = updated | deleted
            for i, bm in enumerate(self.pintsets):
                self.pintsets[i] = bm - to_remove
        if created or updated:
            to_add = created | updated
            ids = np.array(to_add, np.int64)
            values = self.column.loc[to_add]
            bins = np.digitize(values, self.bins)  # type: ignore
            counts = np.bincount(bins)
            for i in np.nonzero(counts)[0]:
                bm = self.pintsets[i]
                selection = bins == i  # boolean mask of values in bin i
                bm.update(ids[selection])  # add them to the PIntSet

    def query(
        self,
        operator_: Callable[[Any, Any], int],
        limit: Any,
        approximate: bool = APPROX,
    ) -> PIntSet:  # blocking...
        """
        Return the list of rows matching the query.
        For example, returning all values less than 10 (< 10) would be
        `query(operator.__lt__, 10)`
        """
        assert self.bins is not None
        pos = np.digitize(limit, self.bins)
        detail = PIntSet()
        if not approximate:
            ids = np.array(self.pintsets[pos], np.int64)
            values = self.column.loc[ids]
            selected = ids[operator_(values, limit)]
            detail.update(selected)

        if operator_ in (operator.lt, operator.le):
            for bm in self.pintsets[:pos]:
                detail.update(bm)
        else:
            for bm in self.pintsets[pos + 1 :]:
                detail.update(bm)
        return detail

    def restricted_query(
        self,
        operator_: Callable[[Any, Any], int],
        limit: Any,
        only_locs: Any,
        approximate: bool = APPROX,
    ) -> PIntSet:  # blocking...
        """
        Returns the subset of only_locs matching the query.
        """
        only_locs = PIntSet.aspintset(only_locs)
        assert self.bins is not None
        pos = np.digitize(limit, self.bins)
        detail = PIntSet()
        if not approximate:
            ids = np.array(self.pintsets[pos] & only_locs, np.int64)
            values = self.column.loc[ids]
            selected = ids[operator_(values, limit)]
            detail.update(selected)

        if operator_ in (operator.lt, operator.le):
            for bm in self.pintsets[:pos]:
                detail.update(bm & only_locs)
        else:
            for bm in self.pintsets[pos + 1 :]:
                detail.update(bm & only_locs)
        return detail

    def range_query(
        self, lower: float, upper: float, all_ids: PIntSet, approximate: bool = APPROX
    ) -> PIntSet:
        """
        Return the PIntSet of all rows with values in range [`lower`, `upper`[
        """
        if lower > upper:
            lower, upper = upper, lower
        assert self.bins is not None
        pos_lo, pos_up = np.digitize([lower, upper], self.bins)
        if pos_up - pos_lo > len(self.bins) // 2:
            exclusion = self.pintsets[: pos_lo + 1] + self.pintsets[pos_up:]
            union = all_ids - PIntSet.union(*exclusion)
        else:
            sets = self.pintsets[pos_lo + 1 : pos_up]
            if not sets:
                return PIntSet()
            union = PIntSet.union(*sets)
        if not approximate:
            detail = PIntSet()
            ids = np.array(self.pintsets[pos_lo], np.int64)
            values = self.column.loc[ids]
            if pos_lo == pos_up:
                selected = ids[(lower <= values) & (values < upper)]
                detail.update(selected)
            else:
                selected = ids[lower <= values]
                detail.update(selected)
                ids = np.array(self.pintsets[pos_up], np.int64)
                values = self.column.loc[ids]
                selected = ids[values < upper]
                detail.update(selected)
            union.update(detail)
        return union

    def range_query_aslist(
        self, lower: float, upper: float, approximate: bool = APPROX
    ) -> List[PIntSet]:
        """
        Return the list of pintsets with values in range [`lower`, `upper`[
        """
        if lower > upper:
            lower, upper = upper, lower
        pos_lo, pos_up = np.digitize([lower, upper], self.bins)  # type: ignore
        detail = PIntSet()
        res = self.pintsets[pos_lo + 1 : pos_up]
        if not approximate:
            ids = np.array(self.pintsets[pos_lo], np.int64)
            values = self.column.loc[ids]
            if pos_lo == pos_up:
                selected = ids[(lower <= values) & (values < upper)]
                detail.update(selected)
            else:
                selected = ids[lower <= values]
                detail.update(selected)
                ids = np.array(self.pintsets[pos_up], np.int64)
                values = self.column.loc[ids]
                selected = ids[values < upper]
                detail.update(selected)
            res.append(detail)
        return res

    def restricted_range_query(
        self, lower: float, upper: float, only_locs: Any, approximate: bool = APPROX
    ) -> PIntSet:
        """
        Return the PIntSet of only_locs rows in range [`lower`, `upper`[
        """
        if lower > upper:
            lower, upper = upper, lower
        only_locs = PIntSet.aspintset(only_locs)
        pos_lo, pos_up = np.digitize([lower, upper], self.bins)  # type: ignore
        sets = [(bm & only_locs) for bm in self.pintsets[pos_lo + 1 : pos_up]]
        if sets:
            union = PIntSet.union(*sets)
        else:
            union = PIntSet()
        if not approximate:
            detail = PIntSet()
            ids = np.array(self.pintsets[pos_lo] & only_locs, np.int64)
            values = self.column.loc[ids]
            if pos_lo == pos_up:
                selected = ids[(lower <= values) & (values < upper)]
                detail.update(selected)
            else:
                selected = ids[lower <= values]
                detail.update(selected)
                ids = np.array(self.pintsets[pos_up] & only_locs, np.int64)
                values = self.column.loc[ids]
                selected = ids[values < upper]
                detail.update(selected)
            union.update(detail)
        return union

    def get_min_bin(self) -> Optional[PIntSet]:
        for bm in self.pintsets:
            if bm:
                return bm
        return None

    def get_max_bin(self) -> Optional[PIntSet]:
        for bm in reversed(self.pintsets):
            if bm:
                return bm
        return None


@document
@def_parameter("bins", np.dtype(int), 126)  # actually 128 with "-inf" and "inf"
@def_parameter("init_threshold", np.dtype(int), 1)
@def_input("table", PTable, hint_type=Sequence[str])
@def_output("result", PTableSelectedView)
@def_output("min_out", PDict, required=False)
@def_output("max_out", PDict, required=False)
class HistogramIndex(Module):
    """
    Compute and maintain an histogram index by dividing the entire range of
    values of a column into a series of intervals (binning).
    Each bin contains the set of indices of rows in the underlying interval
    in order to  provide fast access to these rows.
    Actually `HistogramIndex` is able to build multiple indices, one for each
    column provided in the **table** slot hint.
    """

    def __init__(self, **kwds: Any) -> None:
        super().__init__(
            **kwds
        )
        self._columns: Sequence[str] = []
        self._n_cols = 0
        # will be created when the init_threshold is reached
        self._impl: Dict[str, _HistogramIndexImpl] = {}
        self.selection = PIntSet()  # will be filled when the table is read
        # so realistic initial values for min and max were available
        self.input_module: Optional[Module] = None
        self.input_slot: Optional[str] = None
        self._input_table: Optional[PTable] = None

    def compute_bounds(self, column: str, input_table: PTable) -> Tuple[float, float]:
        values = input_table[column]
        return values.min(), values.max()

    def get_min_bin(self, column: str) -> Optional[PIntSet]:
        if self._impl is None:
            return None
        return self._impl[column].get_min_bin()

    def get_max_bin(self, column: str) -> Optional[PIntSet]:
        if self._impl is None:
            return None
        return self._impl[column].get_max_bin()

    def starting(self) -> None:
        super().starting()
        if self.get_output_slot("min_out"):
            self.min_out = PDict()
        if self.get_output_slot("max_out"):
            self.max_out = PDict()

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        assert input_slot is not None
        steps = 0
        input_table = input_slot.data()
        self._input_table = input_table
        if input_table is None or len(input_table) < self.params.init_threshold:
            # there are not enough rows. it's not worth building an index yet
            return self._return_run_step(self.state_blocked, steps_run=0)
        if not self._columns:
            if (hint := input_slot.hint) is not None:
                self._columns = hint
                self._n_cols = len(hint)
            else:
                raise ProgressiveError("HistogramIndex needs at least one column")
        if not self._impl:
            input_slot.reset()
            input_slot.update(run_number)
            input_slot.clear_buffers()
            assert self._columns is not None
            for column in self._columns:
                bound_min, bound_max = self.compute_bounds(column, input_table)
                self._impl[column] = _HistogramIndexImpl(
                    column, input_table, bound_min, bound_max, self.params.bins
                )
                steps += self.process_min_max(column, input_table)
            self.selection = PIntSet(input_table.index)
            self.result = PTableSelectedView(input_table, self.selection)
            return self._return_run_step(self.state_blocked, len(self.selection))
        else:
            # Many not always, or should the implementation decide?
            for impl in self._impl.values():
                impl.reshape()
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
            steps += indices_len(created) * self._n_cols
            self.selection |= created
        updated: Optional[PIntSet] = None
        if input_slot.updated.any():
            updated = input_slot.updated.next(length=step_size, as_slice=False)
            updated = fix_loc(updated)
            steps += indices_len(updated) * self._n_cols
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        input_table = input_slot.data()
        for col, impl in self._impl.items():
            impl.update_histogram(created, updated, deleted)
            steps += self.process_min_max(col, input_table)
        return self._return_run_step(self.next_state(input_slot), steps_run=steps)

    def process_min_max(self, column: str, input_table: PTable) -> int:
        steps = 0
        if self.min_out is not None:
            min_bin = self.get_min_bin(column)
            if min_bin:
                min_ = input_table[column].loc[min_bin].min()
                self.min_out.update({column: min_})
                steps += len(min_bin)  # TODO: find a better heuristic
        if self.max_out is not None:
            max_bin = self.get_max_bin(column)
            if max_bin:
                max_ = input_table[column].loc[max_bin].max()
                self.max_out.update({column: max_})
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

    def query(
            self,
            column: str,
            operator_: Callable[[Any, Any], Any],
            limit: Any,
            approximate: bool = APPROX,
    ) -> PIntSet:
        """
        Return the list of rows matching the query.
        For example, returning all values less than 10 (< 10) would be
        `query(operator.__lt__, 10)`
        """
        if column in self._impl:
            return self._impl[column].query(operator_, limit, approximate)
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
    ) -> PIntSet:
        """
        Return the list of rows matching the query.
        For example, returning all values less than 10 (< 10) would be
        `query(operator.__lt__, 10)`
        """
        if column in self._impl:
            return self._impl[column].restricted_query(operator_, limit, only_locs, approximate)
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return self._eval_to_ids(column, operator_, limit, only_locs)

    def range_query_aslist(
            self, column: str, lower: float, upper: float, approximate: bool = APPROX
    ) -> List[PIntSet]:
        r"""
        Return the list of rows with values in range \[`lower`, `upper`\[
        """
        if column in self._impl:
            return self._impl[column].range_query_aslist(lower, upper, approximate)
        return []

    def range_query(
            self, column: str, lower: float, upper: float, approximate: bool = APPROX
    ) -> PIntSet:
        r"""
        Return the list of rows with values in range \[`lower`, `upper`\[
        """
        if column in self._impl:
            return self._impl[column].range_query(lower, upper, self.selection, approximate)
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return self._eval_to_ids(
            column,
            operator.__lt__, upper
        ) & self._eval_to_ids(  # optimize later
            column,
            operator.__ge__, lower
        )

    def restricted_range_query(
            self,
            column: str, lower: float,
            upper: float, only_locs: Any, approximate: bool = APPROX
    ) -> PIntSet:
        r"""
        Return the list of rows with values in range \[`lower`, `upper`\[
        among only_locs
        """
        if column in self._impl:
            return self._impl[column].restricted_range_query(
                lower, upper, only_locs, approximate
            )
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return (
            self._eval_to_ids(column, operator.__lt__, upper, only_locs)
            # optimize later
            & self._eval_to_ids(column, operator.__ge__, lower, only_locs)
        )

    def create_dependent_modules(
        self, input_module: Module, input_slot: str, **kwds: Any
    ) -> HistogramIndex:
        with self.grouped():
            self.input_module = input_module
            self.input_slot = input_slot
            hist_index = self
            hist_index.input.table = input_module.output[input_slot]
            # hist_index.input.min = min_.output.result
            # hist_index.input.max = max_.output.result
            return hist_index
