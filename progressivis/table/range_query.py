from __future__ import annotations

import numpy as np

from progressivis.core.module import (
    Module,
    ReturnRunStep,
    def_input,
    def_output,
    def_parameter,
    document
)
from progressivis.core.pintset import PIntSet
from progressivis.core.utils import indices_len
from ..io import Variable
from ..stats import Min, Max
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
    def __init__(self, column: list[str], approximate: bool):
        super(RangeQueryImpl, self).__init__()
        self._table: Optional[BasePTable] = None
        self._column = column
        # self.bins = None
        # self._hist_index = hist_index
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
                lower, upper, approximate=self._approximate
            )
            self.result.assign(new_sel)
            return
        if updated:
            self.result.remove(updated)
            # res = self._eval_to_ids(limit, updated)
            res = hist_index.restricted_range_query(
                lower, upper, only_locs=updated, approximate=self._approximate
            )
            self.result.add(res)
        if created:
            res = hist_index.restricted_range_query(
                lower, upper, only_locs=created, approximate=self._approximate
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
@def_parameter("column", np.dtype(object), "unknown", doc="short description of the **column** parameter")
@def_parameter("watched_key_lower", np.dtype(object), "")
@def_parameter("watched_key_upper", np.dtype(object), "")
@def_input("table", PTable)
@def_input("lower", PDict, required=False, doc="short description of the **lower** input slot")
@def_input("upper", PDict, required=False)
@def_input("min", PDict, required=False)
@def_input("max", PDict, required=False)
@def_input("hist", PTable)
@def_output("result", PTableSelectedView, doc="short description of the **result** output slot")
@def_output("min", PDict, attr_name="_min_table", required=False, doc=(
    "a longer description of the **min** output slot: "
    "Lorem ipsum dolor sit amet, consectetur "
    "adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua."
))
@def_output("max", PDict, attr_name="_max_table", required=False)
class RangeQuery(Module):
    """ """

    def __init__(
        self,
        # hist_index: Optional[HistogramIndex] = None,
        approximate: bool = False,
        **kwds: Any,
    ) -> None:
        super(RangeQuery, self).__init__(**kwds)
        self._impl: RangeQueryImpl = RangeQueryImpl(self.params.column, approximate)
        # self._hist_index: Optional[HistogramIndex] = hist_index
        self._approximate = approximate
        self.default_step_size = 1000
        self.input_module: Optional[Module] = None
        self.hist_index: Optional[HistogramIndex] = None

    # @property
    # def hist_index(self) -> Optional[HistogramIndex]:
    #     return self._hist_index

    # @hist_index.setter
    # def hist_index(self, hi: HistogramIndex) -> None:
    #     self._hist_index = hi
    #     self._impl = RangeQueryImpl(self._column, hi, approximate=self._approximate)
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
        if self.input_module is not None:  # test if already called
            return self
        scheduler = self.scheduler()
        params = self.params
        self.input_module = input_module
        self.input_slot = input_slot
        with scheduler:
            if hist_index is None:
                hist_index = HistogramIndex(
                    column=params.column, group=self.name, scheduler=scheduler
                )
            hist_index.input.table = input_module.output[input_slot]
            if min_ is None:
                min_ = Min(group=self.name, columns=[self.column], scheduler=scheduler)
                min_.input.table = hist_index.output.min_out
            if max_ is None:
                max_ = Max(group=self.name, columns=[self.column], scheduler=scheduler)
                max_.input.table = hist_index.output.max_out
            if min_value is None:
                min_value = Variable(group=self.name, scheduler=scheduler)
                min_value.input.like = min_.output.result

            if max_value is None:
                max_value = Variable(group=self.name, scheduler=scheduler)
                max_value.input.like = max_.output.result

            range_query = self
            range_query.hist_index = hist_index
            range_query.input.hist = hist_index.output.result
            range_query.input.table = input_module.output[input_slot]
            if min_value:
                range_query.input.lower = min_value.output.result
            if max_value:
                range_query.input.upper = max_value.output.result
            range_query.input.min = min_.output.result
            range_query.input.max = max_.output.result

        self.min = min_
        self.max = max_
        self.min_value = min_value
        self.max_value = max_value
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
        #
        # lower/upper
        #
        lower_slot = self.get_input_slot("lower")
        # lower_slot.update(run_number)
        upper_slot = self.get_input_slot("upper")
        limit_changed = False
        if lower_slot.deleted.any():
            lower_slot.deleted.next()
        if lower_slot.updated.any():
            lower_slot.updated.next()
            limit_changed = True
        if lower_slot.created.any():
            lower_slot.created.next()
            limit_changed = True
        if not (lower_slot is upper_slot):
            # upper_slot.update(run_number)
            if upper_slot.deleted.any():
                upper_slot.deleted.next()
            if upper_slot.updated.any():
                upper_slot.updated.next()
                limit_changed = True
            if upper_slot.created.any():
                upper_slot.created.next()
                limit_changed = True
        #
        # min/max
        #
        min_slot = self.get_input_slot("min")
        min_slot.clear_buffers()
        # min_slot.update(run_number)
        # min_slot.created.next()
        # min_slot.updated.next()
        # min_slot.deleted.next()
        max_slot = self.get_input_slot("max")
        max_slot.clear_buffers()
        # max_slot.update(run_number)
        # max_slot.created.next()
        # max_slot.updated.next()
        # max_slot.deleted.next()
        if (
            lower_slot.data() is None
            or upper_slot.data() is None
            or len(lower_slot.data()) == 0
            or len(upper_slot.data()) == 0
        ):
            return self._return_run_step(self.state_blocked, steps_run=0)
        lower_value = lower_slot.data().get(self.watched_key_lower)
        upper_value = upper_slot.data().get(self.watched_key_upper)
        if (
            lower_slot.data() is None
            or upper_slot.data() is None
            or min_slot.data() is None
            or max_slot.data() is None
            or len(min_slot.data()) == 0
            or len(max_slot.data()) == 0
        ):
            return self._return_run_step(self.state_blocked, steps_run=0)
        minv = min_slot.data().get(self.watched_key_lower)
        maxv = max_slot.data().get(self.watched_key_upper)
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
        # input_slot.update(run_number)
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
        hist_slot = self.get_input_slot("hist")
        hist_slot.clear_buffers()
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
