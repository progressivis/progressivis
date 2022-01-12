from __future__ import annotations

import numpy as np

from progressivis.core.module import Module, ReturnRunStep
from progressivis.core.slot import SlotDescriptor
from progressivis.core.bitmap import bitmap
from progressivis.core.utils import indices_len
from ..io import Variable
from ..stats import Min, Max
from ..utils.psdict import PsDict
from . import BaseTable, Table, TableSelectedView
from .module import TableModule
from .hist_index import HistogramIndex

# from .mod_impl import ModuleImpl

from typing import Optional, Any, cast, Iterable


# def _get_physical_table(t):
#     return t if t.base is None else _get_physical_table(t.base)


class _Selection:
    def __init__(self, values: Optional[bitmap] = None):
        self._values = bitmap([]) if values is None else values

    def update(self, values: Iterable[int]) -> None:
        self._values.update(values)

    def remove(self, values: Iterable[int]) -> None:
        self._values = self._values - bitmap(values)

    def assign(self, values: Iterable[int]) -> None:
        self._values = bitmap(values)

    def add(self, values: Iterable[int]) -> None:
        self._values |= bitmap(values)


class RangeQueryImpl:  # (ModuleImpl):
    def __init__(self, column: list[str], approximate: bool):
        super(RangeQueryImpl, self).__init__()
        self._table: Optional[BaseTable] = None
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
        created: Optional[bitmap] = None,
        updated: Optional[bitmap] = None,
        deleted: Optional[bitmap] = None,
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
        table: BaseTable,
        hist_index: HistogramIndex,
        lower: float,
        upper: float,
        limit_changed: bool,
        created: Optional[bitmap] = None,
        updated: Optional[bitmap] = None,
        deleted: Optional[bitmap] = None,
    ) -> None:
        self._table = table
        self.result = _Selection()
        self.is_started = True
        self.resume(hist_index, lower, upper, limit_changed, created, updated, deleted)


class RangeQuery(TableModule):
    """ """

    parameters = [
        ("column", np.dtype(object), "unknown"),
        ("watched_key_lower", np.dtype(object), ""),
        ("watched_key_upper", np.dtype(object), ""),
        # ('hist_index', object, None) # to improve ...
    ]
    inputs = [
        SlotDescriptor("table", type=Table, required=True),
        SlotDescriptor("lower", type=Table, required=False),
        SlotDescriptor("upper", type=Table, required=False),
        SlotDescriptor("min", type=PsDict, required=False),
        SlotDescriptor("max", type=PsDict, required=False),
        SlotDescriptor("hist", type=Table, required=True),
    ]
    outputs = [
        SlotDescriptor("min", type=Table, required=False),
        SlotDescriptor("max", type=Table, required=False),
    ]

    def __init__(
        self,
        # hist_index: Optional[HistogramIndex] = None,
        approximate: bool = False,
        **kwds: Any
    ) -> None:
        super(RangeQuery, self).__init__(**kwds)
        self._impl: RangeQueryImpl = RangeQueryImpl(self.params.column, approximate)
        # self._hist_index: Optional[HistogramIndex] = hist_index
        self._approximate = approximate
        self._column = self.params.column
        self._watched_key_lower = self.params.watched_key_lower
        if not self._watched_key_lower:
            self._watched_key_lower = self._column
        self._watched_key_upper = self.params.watched_key_upper
        if not self._watched_key_upper:
            self._watched_key_upper = self._column
        self.default_step_size = 1000
        self.input_module: Optional[Module] = None
        self._min_table: Optional[PsDict] = None
        self._max_table: Optional[PsDict] = None
        self.hist_index: Optional[HistogramIndex] = None

    # @property
    # def hist_index(self) -> Optional[HistogramIndex]:
    #     return self._hist_index

    # @hist_index.setter
    # def hist_index(self, hi: HistogramIndex) -> None:
    #     self._hist_index = hi
    #     self._impl = RangeQueryImpl(self._column, hi, approximate=self._approximate)

    def create_dependent_modules(
        self,
        input_module: Module,
        input_slot: str,
        min_: Optional[Module] = None,
        max_: Optional[Module] = None,
        min_value: Optional[Module] = None,
        max_value: Optional[Module] = None,
        **kwds: Any
    ) -> RangeQuery:
        if self.input_module is not None:  # test if already called
            return self
        with self.grouped():
            scheduler = self.scheduler()
            params = self.params
            self.input_module = input_module
            self.input_slot = input_slot
            with scheduler:
                hist_index = HistogramIndex(
                    column=params.column, group=self.name, scheduler=scheduler
                )
                hist_index.input.table = input_module.output[input_slot]
                if min_ is None:
                    min_ = Min(
                        group=self.name, columns=[self._column], scheduler=scheduler
                    )
                    min_.input.table = hist_index.output.min_out
                if max_ is None:
                    max_ = Max(
                        group=self.name, columns=[self._column], scheduler=scheduler
                    )
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
            self._min_table = PsDict({self._column: np.inf})
        if self._max_table is None:
            self._max_table = PsDict({self._column: -np.inf})

    def _set_minmax_out(self, attr_: str, val: float) -> None:
        d = {self._column: val}
        if getattr(self, attr_) is None:
            setattr(self, attr_, PsDict(d))
        else:
            getattr(self, attr_).update(d)

    def _set_min_out(self, val: float) -> None:
        return self._set_minmax_out("_min_table", val)

    def _set_max_out(self, val: float) -> None:
        return self._set_minmax_out("_max_table", val)

    def get_data(self, name: str) -> Any:
        if name == "min":
            return self._min_table
        if name == "max":
            return self._max_table
        return super(RangeQuery, self).get_data(name)

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
        lower_value = lower_slot.data().get(self._watched_key_lower)
        upper_value = upper_slot.data().get(self._watched_key_upper)
        if (
            lower_slot.data() is None
            or upper_slot.data() is None
            or min_slot.data() is None
            or max_slot.data() is None
            or len(min_slot.data()) == 0
            or len(max_slot.data()) == 0
        ):
            return self._return_run_step(self.state_blocked, steps_run=0)
        minv = min_slot.data().get(self._watched_key_lower)
        maxv = max_slot.data().get(self._watched_key_upper)
        if (
            lower_value is None
            or np.isnan(lower_value)
            or lower_value < minv
            or lower_value >= maxv
        ):
            lower_value = minv
            limit_changed = True
        if (
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
        deleted: Optional[bitmap] = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(length=step_size, as_slice=False)
            steps += indices_len(deleted)
        created: Optional[bitmap] = None
        if input_slot.created.any():
            created = input_slot.created.next(length=step_size, as_slice=False)
            steps += indices_len(created)
        updated: Optional[bitmap] = None
        if input_slot.updated.any():
            updated = input_slot.updated.next(length=step_size, as_slice=False)
            steps += indices_len(updated)
        input_table = input_slot.data()
        if not self.result:
            self.result = TableSelectedView(input_table, bitmap([]))
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
        self.selected.selection = self._impl.result._values
        return self._return_run_step(self.next_state(input_slot), steps)
