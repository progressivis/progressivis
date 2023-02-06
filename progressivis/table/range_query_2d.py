from __future__ import annotations

import numpy as np

import itertools as it

from ..core.module import Module, ReturnRunStep, def_input, def_output, def_parameter
from ..core.pintset import PIntSet
from ..core.utils import indices_len
from ..utils.psdict import PDict
from ..io import Variable
from ..stats import Min, Max
from .table_base import PTableSelectedView, BasePTable
from .table import PTable
from .hist_index import HistogramIndex
from .merge_dict import MergeDict

from typing import Optional, Any, cast, Union, Iterable


class _Selection(object):
    def __init__(self, values: Optional[PIntSet] = None) -> None:
        self._values = PIntSet([]) if values is None else values

    def update(self, values: Iterable[int]) -> None:
        self._values.update(values)

    def remove(self, values: Iterable[int]) -> None:
        self._values = self._values - PIntSet(values)

    def assign(self, values: Iterable[int]) -> None:
        self._values = PIntSet(values)


class RangeQuery2dImpl:  # (ModuleImpl):
    def __init__(
        self,
        column_x: str,
        column_y: str,
        # hist_index_x: HistogramIndex,
        # hist_index_y: HistogramIndex,
        approximate: bool,
    ) -> None:
        super(RangeQuery2dImpl, self).__init__()
        self._table: Optional[BasePTable] = None
        self._column_x = column_x
        self._column_y = column_y
        # self.bins = None
        # self._hist_index_x = hist_index_x
        # self._hist_index_y = hist_index_y
        self._approximate = approximate
        self.result: Optional[_Selection] = None
        self.is_started = False

    def resume(
        self,
        hist_index_x: HistogramIndex,
        hist_index_y: HistogramIndex,
        lower_x: float,
        upper_x: float,
        lower_y: float,
        upper_y: float,
        limit_changed: bool,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
    ) -> None:
        assert self.result
        if limit_changed:
            new_sel_x = hist_index_x.range_query_aslist(
                lower_x, upper_x, approximate=self._approximate
            )
            new_sel_y = hist_index_y.range_query_aslist(
                lower_y, upper_y, approximate=self._approximate
            )
            if new_sel_x is None or new_sel_y is None:
                new_sel_x = hist_index_x.range_query(
                    lower_x, upper_x, approximate=self._approximate
                )
                new_sel_y = hist_index_y.range_query(
                    lower_y, upper_y, approximate=self._approximate
                )
                new_sel = new_sel_x & new_sel_y
            else:
                new_sel = PIntSet.union(
                    *(x & y for x, y in it.product(new_sel_x, new_sel_y))
                )
            self.result.assign(new_sel)
            return
        if updated:
            self.result.remove(updated)
            res_x = hist_index_x.restricted_range_query(
                lower_x, upper_x, only_locs=updated, approximate=self._approximate
            )
            res_y = hist_index_y.restricted_range_query(
                lower_y, upper_y, only_locs=updated, approximate=self._approximate
            )
            self.result.update(res_x & res_y)
        if created:
            res_x = hist_index_x.restricted_range_query(
                lower_x, upper_x, only_locs=created, approximate=self._approximate
            )
            res_y = hist_index_y.restricted_range_query(
                lower_y, upper_y, only_locs=created, approximate=self._approximate
            )
            self.result.update(res_x & res_y)
        if deleted:
            self.result.remove(deleted)

    def start(
        self,
        table: BasePTable,
        hist_index_x: HistogramIndex,
        hist_index_y: HistogramIndex,
        lower_x: float,
        upper_x: float,
        lower_y: float,
        upper_y: float,
        limit_changed: bool,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
    ) -> None:
        self.result = _Selection()
        self._table = table
        self.is_started = True
        return self.resume(
            hist_index_x,
            hist_index_y,
            lower_x,
            upper_x,
            lower_y,
            upper_y,
            limit_changed,
            created,
            updated,
            deleted,
        )


@def_parameter("column_x", np.dtype(object), "unknown")
@def_parameter("column_y", np.dtype(object), "unknown")
@def_parameter("watched_key_lower_x", np.dtype(object), "")
@def_parameter("watched_key_upper_x", np.dtype(object), "")
@def_parameter("watched_key_lower_y", np.dtype(object), "")
@def_parameter("watched_key_upper_y", np.dtype(object), "")
@def_input("table", PTable)
@def_input("lower", PDict, required=False)
@def_input("upper", PDict, required=False)
@def_input("min", PDict, required=False)
@def_input("max", PDict, required=False)
@def_input("hist_x", PTable)
@def_input("hist_y", PTable)
@def_output("result", PTableSelectedView)
@def_output("min", PDict, attr_name="_min_table", required=False)
@def_output("max", PDict, attr_name="_max_table", required=False)
class RangeQuery2d(Module):
    """ """

    def __init__(
        self,
        # hist_index: Optional[HistogramIndex] = None,
        approximate: bool = False,
        **kwds: Any,
    ) -> None:
        super(RangeQuery2d, self).__init__(**kwds)
        # self._hist_index_x: Optional[HistogramIndex] = None
        # self._hist_index_y: Optional[HistogramIndex] = None
        self._approximate = approximate
        self._column_x: str = self.params.column_x
        self._column_y: str = self.params.column_y
        self._impl = RangeQuery2dImpl(self._column_x, self._column_y, approximate)
        # X ...
        self._watched_key_lower_x = self.params.watched_key_lower_x
        if not self._watched_key_lower_x:
            self._watched_key_lower_x = self._column_x
        self._watched_key_upper_x = self.params.watched_key_upper_x
        if not self._watched_key_upper_x:
            self._watched_key_upper_x = self._column_x
        # Y ...
        self._watched_key_lower_y = self.params.watched_key_lower_y
        if not self._watched_key_lower_y:
            self._watched_key_lower_y = self._column_y
        self._watched_key_upper_y = self.params.watched_key_upper_y
        if not self._watched_key_upper_y:
            self._watched_key_upper_y = self._column_y
        self.default_step_size = 1000
        self.input_module: Optional[Module] = None
        self._min_table: Optional[PDict]
        self._max_table: Optional[PDict]

    def create_dependent_modules(
        self,
        input_module: Module,
        input_slot: str,
        min_: Optional[Module] = None,
        max_: Optional[Module] = None,
        min_value: Union[None, bool, Module] = None,
        max_value: Union[None, bool, Module] = None,
        **kwds: Any,
    ) -> RangeQuery2d:
        """
        Beware, {min,max}_value=None is not the same as {min,max}_value=False.
        With None, a min module is created and connected.
        With False, it is not created and not connected.
        """
        if self.input_module is not None:  # test if already called
            return self
        with self.grouped():
            scheduler = self.scheduler()
            params = self.params
            self.input_module = input_module
            self.input_slot = input_slot
            with scheduler:
                hist_index_x = HistogramIndex(
                    column=params.column_x, group=self.name, scheduler=scheduler
                )
                self.hist_index_x = hist_index_x
                hist_index_x.input.table = input_module.output[input_slot]
                hist_index_y = HistogramIndex(
                    column=params.column_y, group=self.name, scheduler=scheduler
                )
                self.hist_index_y = hist_index_y
                hist_index_y.input.table = input_module.output[input_slot]
                if min_ is None:
                    min_x = Min(
                        group=self.name, scheduler=scheduler, columns=[self._column_x]
                    )
                    min_x.input.table = hist_index_x.output.min_out
                    min_y = Min(
                        group=self.name, scheduler=scheduler, columns=[self._column_y]
                    )
                    min_y.input.table = hist_index_y.output.min_out
                    min_ = MergeDict(group=self.name, scheduler=scheduler)
                    min_.input.first = min_x.output.result
                    min_.input.second = min_y.output.result
                if max_ is None:
                    max_x = Max(
                        group=self.name, scheduler=scheduler, columns=[self._column_x]
                    )
                    max_x.input.table = hist_index_x.output.max_out
                    max_y = Max(
                        group=self.name, scheduler=scheduler, columns=[self._column_y]
                    )
                    max_y.input.table = hist_index_y.output.max_out
                    max_ = MergeDict(group=self.name, scheduler=scheduler)
                    max_.input.first = max_x.output.result
                    max_.input.second = max_y.output.result
                if min_value is None:
                    min_value = Variable(group=self.name, scheduler=scheduler)
                    min_value.input.like = min_.output.result
                if max_value is None:
                    max_value = Variable(group=self.name, scheduler=scheduler)
                    max_value.input.like = max_.output.result

                range_query = self
                range_query.input.hist_x = hist_index_x.output.result
                range_query.input.hist_y = hist_index_y.output.result
                # self._impl = RangeQuery2dImpl(
                #     self._column_x,
                #     self._column_y,
                #     hist_index_x,
                #     hist_index_y,
                #     approximate=self._approximate,
                # )
                range_query.input.table = input_module.output[
                    input_slot
                ]  # hist_index_x.output.result
                if min_value:
                    assert isinstance(min_value, Module)
                    range_query.input.lower = min_value.output.result
                if max_value:
                    assert isinstance(max_value, Module)
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
            self._min_table = PDict({self._column_x: np.inf, self._column_y: np.inf})
            # PTable(name=None, dshape=self.min_max_dshape)
        if self._max_table is None:
            self._max_table = PDict({self._column_x: -np.inf, self._column_y: -np.inf})
            # PTable(name=None, dshape=self.min_max_dshape)

    def _set_minmax_out(self, attr_: str, val_x: float, val_y: float) -> None:
        d = {self._column_x: val_x, self._column_y: val_y}
        if getattr(self, attr_) is None:
            setattr(self, attr_, PDict(d))
            # PTable(name=None, dshape=self.min_max_dshape)
        else:
            getattr(self, attr_).update(d)

    def _set_min_out(self, val_x: float, val_y: float) -> None:
        return self._set_minmax_out("_min_table", val_x, val_y)

    def _set_max_out(self, val_x: float, val_y: float) -> None:
        return self._set_minmax_out("_max_table", val_x, val_y)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        # input_slot.update(run_number)
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
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        assert isinstance(input_table, BasePTable)
        if self.result is None:
            self.result = PTableSelectedView(input_table, PIntSet([]))
        self._create_min_max()
        # param = self.params
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
        # X ...
        lower_value_x = lower_slot.data().get(self._watched_key_lower_x)
        upper_value_x = upper_slot.data().get(self._watched_key_upper_x)
        # Y ...
        lower_value_y = lower_slot.data().get(self._watched_key_lower_y)
        upper_value_y = upper_slot.data().get(self._watched_key_upper_y)
        if (
            lower_slot.data() is None
            or upper_slot.data() is None
            or len(min_slot.data()) == 0
            or len(max_slot.data()) == 0
        ):
            return self._return_run_step(self.state_blocked, steps_run=0)
        # X ...
        minv_x = min_slot.data().get(self._watched_key_lower_x)
        maxv_x = max_slot.data().get(self._watched_key_upper_x)
        # Y ...
        minv_y = min_slot.data().get(self._watched_key_lower_y)
        maxv_y = max_slot.data().get(self._watched_key_upper_y)
        # X ...
        if (
            lower_value_x is None
            or np.isnan(lower_value_x)
            or lower_value_x < minv_x
            or lower_value_x >= maxv_x
        ):
            lower_value_x = minv_x
            limit_changed = True
        if (
            upper_value_x is None
            or np.isnan(upper_value_x)
            or upper_value_x > maxv_x
            or upper_value_x <= minv_x
            or upper_value_x <= lower_value_x
        ):
            upper_value_x = maxv_x
            limit_changed = True
        # Y ...
        if (
            lower_value_y is None
            or np.isnan(lower_value_y)
            or lower_value_y < minv_y
            or lower_value_y >= maxv_y
        ):
            lower_value_y = minv_y
            limit_changed = True
        if (
            upper_value_y is None
            or np.isnan(upper_value_y)
            or upper_value_y > maxv_y
            or upper_value_y <= minv_y
            or upper_value_y <= lower_value_y
        ):
            upper_value_y = maxv_y
            limit_changed = True
        self._set_min_out(lower_value_x, lower_value_y)
        self._set_max_out(upper_value_x, upper_value_y)
        if steps == 0 and not limit_changed:
            return self._return_run_step(self.state_blocked, steps_run=0)
        # ...
        hist_x_slot = self.get_input_slot("hist_x")
        hist_x_slot.clear_buffers()
        hist_y_slot = self.get_input_slot("hist_y")
        hist_y_slot.clear_buffers()

        assert self._impl
        if not self._impl.is_started:
            self._impl.start(
                input_table,
                cast(HistogramIndex, hist_x_slot.output_module),
                cast(HistogramIndex, hist_y_slot.output_module),
                lower_value_x,
                upper_value_x,
                lower_value_y,
                upper_value_y,
                limit_changed,
                created=created,
                updated=updated,
                deleted=deleted,
            )
            assert self._impl.result
            self.result.selection = self._impl.result._values
        else:
            self._impl.resume(
                cast(HistogramIndex, hist_x_slot.output_module),
                cast(HistogramIndex, hist_y_slot.output_module),
                lower_value_x,
                upper_value_x,
                lower_value_y,
                upper_value_y,
                limit_changed,
                created=created,
                updated=updated,
                deleted=deleted,
            )
            assert self._impl.result
            self.result.selection = self._impl.result._values
        return self._return_run_step(self.next_state(input_slot), steps)
