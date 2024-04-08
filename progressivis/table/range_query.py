from __future__ import annotations

import numpy as np
from progressivis.core.module import (
    Module,
    ReturnRunStep,
    def_input,
    def_output,
    def_parameter,
    document,
)
from progressivis.core.pintset import PIntSet
from progressivis.core.utils import indices_len
from ..io import Variable
from ..utils.psdict import PDict
from . import BasePTable, PTable, PTableSelectedView
from .hist_index import HistogramIndex

# from .mod_impl import ModuleImpl

from typing import Optional, Any, cast, Iterable


# def _get_physical_table(t):
#     return t if t.base is None else _get_physical_table(t.base)


class _Selection:
    def __init__(self, values: Optional[PIntSet] = None):
        self._values = PIntSet([]) if values is None else values

    def update(self, values: Iterable[int]) -> None:
        self._values.update(values)

    def remove(self, values: Iterable[int]) -> None:
        self._values = self._values - PIntSet(values)

    def assign(self, values: Iterable[int]) -> None:
        self._values = PIntSet(values)

    def add(self, values: Iterable[int]) -> None:
        self._values |= PIntSet(values)


class RangeQueryImpl:  # (ModuleImpl):
    def __init__(self, column: str, approximate: bool):
        super(RangeQueryImpl, self).__init__()
        self._table: Optional[BasePTable] = None
        self._column = column
        self._approximate = approximate
        self.result: Optional[_Selection] = None
        self.is_started = False

    def resume(
        self,
        hist_index: HistogramIndex,
        lower: float,
        upper: float,
        limit_changed: bool,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
    ) -> None:
        assert self.result
        if limit_changed:
            new_sel = hist_index.range_query(
                self._column, lower, upper, approximate=self._approximate
            )
            self.result.assign(new_sel)
            return
        if updated:
            self.result.remove(updated)
            res = hist_index.restricted_range_query(
                self._column,
                lower,
                upper,
                only_locs=updated,
                approximate=self._approximate,
            )
            self.result.add(res)
        if created:
            res = hist_index.restricted_range_query(
                self._column,
                lower,
                upper,
                only_locs=created,
                approximate=self._approximate,
            )
            self.result.update(res)
        if deleted:
            self.result.remove(deleted)

    def start(
        self,
        table: BasePTable,
        hist_index: HistogramIndex,
        lower: float,
        upper: float,
        limit_changed: bool,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
    ) -> None:
        self._table = table
        self.result = _Selection()
        self.is_started = True
        self.resume(hist_index, lower, upper, limit_changed, created, updated, deleted)


@document
@def_parameter(
    "column",
    np.dtype(object),
    "unknown",
    doc=(
        "The column in the **table** input slot concerned by the query. "
        "This parameter is mandatory"
    ),
)
@def_parameter(
    "watched_key_lower",
    np.dtype(object),
    "",
    doc=(
        "The key in the **lower** input slot (which is a **PDict**) "
        "giving the lower bound of the query. "
        'When unset (i.e. ==""), the **column** parameter is used instead.'
    ),
)
@def_parameter(
    "watched_key_upper",
    np.dtype(object),
    "",
    doc=(
        "The key in the **upper** input slot (which is a **PDict**) "
        "giving the upper bound of the query. "
        'When unset (i.e. ==""), the **column** parameter is used instead.'
    ),
)
@def_input("table", PTable, doc="Provides data to be queried.")
@def_input(
    "lower",
    PDict,
    doc=(
        "Provides a **PDict** object containing the lower bound of the query. "
        "The key giving the bound is set by the **watched_key_lower** parameter "
        "when it is different from the **column** parameter."
    ),
)
@def_input(
    "upper",
    PDict,
    doc=(
        "Provides a **PDict** object containing the upper bound of the query. "
        "The key giving the bound is set by the **watched_key_upper** parameter "
        "when it is different from the **column** parameter."
    ),
)
@def_input(
    "min",
    PDict,
    doc=(
        "The minimum value in the input data. This mandatory parameter could be provided "
        "by the `create_dependent_modules()` method."
    ),
)
@def_input(
    "max",
    PDict,
    doc=(
        "The maximum value in the input data. This mandatory parameter could be provided"
        "by the `create_dependent_modules()` method."
    ),
)
@def_input(
    "hist",
    PTable,
    doc=(
        "**HistogramIndex** module output connected to the same input/column."
        "This mandatory parameter could be provided "
        "by the `create_dependent_modules()` method."
    ),
)
@def_output("result", PTableSelectedView, doc="Query main result")
@def_output(
    "min", PDict, attr_name="_min_table", required=False, doc="min value of output data"
)
@def_output(
    "max", PDict, attr_name="_max_table", required=False, doc="max value of output data"
)
class RangeQuery(Module):
    """
    Selects rows that contain values within a provided range along a given axis (column)
    """

    def __init__(
        self,
        approximate: bool = False,
        quantiles: bool = False,
        **kwds: Any,
    ) -> None:
        super(RangeQuery, self).__init__(**kwds)
        self._impl: RangeQueryImpl = RangeQueryImpl(self.params.column, approximate)
        self._approximate = approximate
        self.default_step_size = 1000
        self._quantiles = quantiles
        self.input_module: Optional[Module] = None
        self.hist_index: Optional[HistogramIndex] = None

    @property
    def column(self) -> str:
        return str(self.params.column)

    @property
    def watched_key_lower(self) -> str:
        return self.params.watched_key_lower or self.column

    @property
    def watched_key_upper(self) -> str:
        return self.params.watched_key_upper or self.column

    def create_dependent_modules(
        self,
        input_module: Module,
        input_slot: str,
        min_: Optional[Module] = None,
        max_: Optional[Module] = None,
        min_value: Optional[Module] = None,
        max_value: Optional[Module] = None,
        hist_index: Optional[HistogramIndex] = None,
        **kwds: Any,
    ) -> RangeQuery:
        """
        Creates a default configuration containing the necessary underlying modules.
        Beware, {min,max}_value=None is not the same as {min,max}_value=False.
        With None, a min module is created and connected.
        With False, it is not created and not connected.

        """
        if self.input_module is not None:  # test if already called
            return self
        scheduler = self.scheduler()
        params = self.params
        self.input_module = input_module
        self.input_slot = input_slot
        with scheduler:
            if hist_index is None:
                hist_index = HistogramIndex(
                    group=self.name, scheduler=scheduler
                )
            hist_index.input.table = input_module.output[input_slot][params.column,]
            if min_value is None:
                assert hasattr(min_, "result") or min_ is None
                init_min = min_.result if min_ is not None else hist_index.min_out
                min_value = Variable(init_min, group=self.name, scheduler=scheduler)
            if max_value is None:
                assert hasattr(max_, "result") or max_ is None
                init_max = max_.result if max_ is not None else hist_index.max_out
                max_value = Variable(init_max, group=self.name, scheduler=scheduler)
            range_query = self
            range_query.dep.hist_index = hist_index
            range_query.input.hist = hist_index.output.result
            range_query.input.table = hist_index.output.result
            if min_value:
                range_query.input.lower = min_value.output.result
            if max_value:
                range_query.input.upper = max_value.output.result
            if min_:
                range_query.input.min = min_.output.result
            else:
                range_query.input.min = hist_index.output.min_out
            if max_:
                range_query.input.max = max_.output.result
            else:
                range_query.input.max = hist_index.output.max_out

        self.dep.min = min_
        self.dep.max = max_
        self.dep.hist_index = hist_index
        self.dep.min_value = min_value
        self.dep.max_value = max_value
        return range_query

    def _create_min_max(self) -> None:
        if self._min_table is None:
            self._min_table = PDict({self.column: np.inf})
        if self._max_table is None:
            self._max_table = PDict({self.column: -np.inf})

    def _set_minmax_out(self, attr_: str, val: float) -> None:
        d = {self.column: val}
        if getattr(self, attr_) is None:
            setattr(self, attr_, PDict(d))
        else:
            getattr(self, attr_).update(d)

    def _set_min_out(self, val: float) -> None:
        return self._set_minmax_out("_min_table", val)

    def _set_max_out(self, val: float) -> None:
        return self._set_minmax_out("_max_table", val)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        self._create_min_max()
        hist_slot = self.get_input_slot("hist")
        hist_slot.clear_buffers()
        #
        # lower/upper
        #
        lower_slot = self.get_input_slot("lower")
        upper_slot = self.get_input_slot("upper")
        limit_changed = False
        if not self._quantiles and (lower_slot.has_buffered() or upper_slot.has_buffered()):
            limit_changed = True
        lower_slot.clear_buffers()
        upper_slot.clear_buffers()
        #
        # min/max
        #
        min_slot = self.get_input_slot("min")
        min_slot.clear_buffers()
        max_slot = self.get_input_slot("max")
        max_slot.clear_buffers()
        if not (
            lower_slot.has_data()
            and upper_slot.has_data()
            and min_slot.has_data()
            and max_slot.has_data()
        ):
            return self._return_run_step(self.state_blocked, steps_run=0)
        lower_value = lower_slot.data().get(self.watched_key_lower)
        upper_value = upper_slot.data().get(self.watched_key_upper)
        minv = min_slot.data().get(self.watched_key_lower)
        if minv is None:  # watched key could be defined only for lower/upper bounds
            minv = min_slot.data().get(self.column)
        maxv = max_slot.data().get(self.watched_key_upper)
        if maxv is None:
            maxv = max_slot.data().get(self.column)
        if lower_value == "*":
            lower_value = minv
        elif (
            lower_value is None
            or np.isnan(lower_value)
            or lower_value < minv
            or lower_value >= maxv
        ):
            lower_value = minv
            limit_changed = True
        if upper_value == "*":
            upper_value = maxv
        elif (
            upper_value is None
            or np.isnan(upper_value)
            or upper_value > maxv
            or upper_value <= minv
            or upper_value <= lower_value
        ):
            upper_value = maxv
            limit_changed = True
        self._set_min_out(lower_value)
        self._set_max_out(upper_value)
        if not input_slot.has_buffered() and not limit_changed:
            return self._return_run_step(self.state_blocked, steps_run=0)
        # ...
        steps = 0
        deleted: Optional[PIntSet] = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(length=step_size, as_slice=False)
            steps += indices_len(deleted)
        created: Optional[PIntSet] = None
        if input_slot.created.any():
            created = input_slot.created.next(length=step_size, as_slice=False)
            steps += indices_len(created)
        updated: Optional[PIntSet] = None
        if input_slot.updated.any():
            updated = input_slot.updated.next(length=step_size, as_slice=False)
            steps += indices_len(updated)
        input_table = input_slot.data()
        if self.result is None:
            self.result = PTableSelectedView(input_table, PIntSet([]))
        assert self._impl
        if not self._impl.is_started:
            self._impl.start(
                input_table,
                cast(HistogramIndex, hist_slot.output_module),
                lower_value,
                upper_value,
                limit_changed,
                created=created,
                updated=updated,
                deleted=deleted,
            )
        else:
            self._impl.resume(
                cast(HistogramIndex, hist_slot.output_module),
                lower_value,
                upper_value,
                limit_changed,
                created=created,
                updated=updated,
                deleted=deleted,
            )
        assert self._impl.result
        self.result.selection = self._impl.result._values
        return self._return_run_step(self.next_state(input_slot), steps)
