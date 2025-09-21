from __future__ import annotations

import numpy as np

import itertools as it
from progressivis.core.api import (
    Module,
    def_input,
    def_output,
    def_parameter,
    document,
    PIntSet,
    indices_len,
    Slot,
    ReturnRunStep,
)
from progressivis.utils.api import PDict
from progressivis.table.api import (
    PTableSelectedView,
    BasePTable,
    PTable,
)

from .binning_index_nd import BinningIndexND

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


class RangeQuery2DImpl:  # (ModuleImpl):
    def __init__(self, approximate: bool, column_x: str, column_y: str) -> None:
        super().__init__()
        self._table: Optional[BasePTable] = None
        self._approximate = approximate
        self._column_x: str = column_x
        self._column_y: str = column_y
        self.result: Optional[_Selection] = None
        self.is_started = False

    def resume(
        self,
        index_2d: BinningIndexND,
        lower_x: float,
        upper_x: float,
        lower_y: float,
        upper_y: float,
        limit_changed: bool,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
        only_bins_x: PIntSet = PIntSet(),
        only_bins_y: PIntSet = PIntSet(),
    ) -> None:
        assert self.result
        col_x = self._column_x
        col_y = self._column_y
        if limit_changed:
            new_sel_x = index_2d.range_query_asgen(
                col_x, lower_x, upper_x, approximate=self._approximate
            )
            new_sel_y = index_2d.range_query_asgen(
                col_y, lower_y, upper_y, approximate=self._approximate
            )
            if new_sel_x is None or new_sel_y is None:
                new_sel_x = index_2d.range_query(
                    col_x, lower_x, upper_x, approximate=self._approximate
                )
                new_sel_y = index_2d.range_query(
                    col_y, lower_y, upper_y, approximate=self._approximate
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
            res_x = index_2d.restricted_range_query(
                col_x,
                lower_x,
                upper_x,
                only_locs=updated,
                approximate=self._approximate,
                only_bins=only_bins_x,
            )
            res_y = index_2d.restricted_range_query(
                col_y,
                lower_y,
                upper_y,
                only_locs=updated,
                approximate=self._approximate,
                only_bins=only_bins_y,
            )
            self.result.update(res_x & res_y)
        if created:
            res_x = index_2d.restricted_range_query(
                col_x,
                lower_x,
                upper_x,
                only_locs=created,
                approximate=self._approximate,
                only_bins=only_bins_x,
            )
            res_y = index_2d.restricted_range_query(
                col_y,
                lower_y,
                upper_y,
                only_locs=created,
                approximate=self._approximate,
                only_bins=only_bins_y,
            )
            self.result.update(res_x & res_y)
        if deleted:
            self.result.remove(deleted)

    def start(
        self,
        table: BasePTable,
        index_2d: BinningIndexND,
        lower_x: float,
        upper_x: float,
        lower_y: float,
        upper_y: float,
        limit_changed: bool,
        created: Optional[PIntSet] = None,
        updated: Optional[PIntSet] = None,
        deleted: Optional[PIntSet] = None,
        only_bins_x: PIntSet = PIntSet(),
        only_bins_y: PIntSet = PIntSet(),
    ) -> None:
        self.result = _Selection()
        self._table = table
        self.is_started = True
        return self.resume(
            index_2d,
            lower_x,
            upper_x,
            lower_y,
            upper_y,
            limit_changed,
            created,
            updated,
            deleted,
            only_bins_x,
            only_bins_y,
        )


@document
@def_parameter(
    "column_x",
    np.dtype(object),
    "unknown",
    doc=(
        "The **x axis** column in the **table**"
        " input slot concerned by the query. "
        "This parameter is mandatory"
    ),
)
@def_parameter(
    "column_y",
    np.dtype(object),
    "unknown",
    doc=(
        "The **y axis** column in the **table** "
        "input slot concerned by the query. "
        "This parameter is mandatory"
    ),
)
@def_parameter(
    "watched_key_lower_x",
    np.dtype(object),
    "",
    doc=(
        "The **x axis** key in the **lower** input slot (which is a **PDict**)"
        " giving the lower bound of the query. "
        'When unset (i.e. ==""), the **column** parameter is used instead.'
    ),
)
@def_parameter(
    "watched_key_upper_x",
    np.dtype(object),
    "",
    doc=(
        "The **x axis** key in the **upper** input slot (which is a **PDict**)"
        " giving the upper bound of the query. "
        'When unset (i.e. ==""), the **column** parameter is used instead.'
    ),
)
@def_parameter(
    "watched_key_lower_y",
    np.dtype(object),
    "",
    doc=(
        "The **y axis** key in the **lower** input slot (which is a **PDict**)"
        " giving the lower bound of the query. "
        'When unset (i.e. ==""), the **column** parameter is used instead.'
    ),
)
@def_parameter(
    "watched_key_upper_y",
    np.dtype(object),
    "",
    doc=(
        "The **y axis** key in the **upper** input slot (which is a **PDict**)"
        " giving the upper bound of the query. "
        'When unset (i.e. ==""), the **column** parameter is used instead.'
    ),
)
# @def_input("table", PTable, doc="Provides data to be queried.")
@def_input(
    "lower",
    PDict,
    required=False,
    doc=(
        "Provides a **PDict** object containing the 2D lower bound of the query. "
        "The x, y axis keys giving the bound are set by the **watched_key_lower_{x|}**"
        " parameters when they are different from the **column_{x|y}** parameters."
    ),
)
@def_input(
    "upper",
    PDict,
    required=False,
    doc=(
        "Provides a **PDict** object containing the 2D upper bound of the query. "
        "The x, y axis keys giving the bound are set by the **watched_key_upper_{x|}**"
        " parameters when they are different from the **column_{x|y}** parameters."
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
    "timestamps_x",
    PDict,
    doc=("Gives information about bins changed between 2 run steps on the `x` axis"),
    required=False,
)
@def_input(
    "timestamps_y",
    PDict,
    doc=("Gives information about bins changed between 2 run steps on the `y` axis"),
    required=False,
)
@def_input(
    "index",
    PTable,
    doc=(
        "**BinningIndexND** module output connected to the `x` filtering input/column."
        "This mandatory parameter could be provided "
        "by the `create_dependent_modules()` method."
    ),
)
@def_output("result", PTableSelectedView)
@def_output("min", PDict, attr_name="_min_table", required=False, doc="min doc")
@def_output("max", PDict, attr_name="_max_table", required=False)
class RangeQuery2D(Module):
    """
    Selects rows that contain values within a provided 2D interval (two columns)
    """

    def __init__(
        self,
        approximate: bool = False,
        **kwds: Any,
    ) -> None:
        """
        Parameters
        ----------
        approximate:
            approx ...
        kwds:
            keywords
        """
        super().__init__(**kwds)
        self._approximate = approximate
        self._column_x: str = self.params.column_x
        self._column_y: str = self.params.column_y
        self._impl = RangeQuery2DImpl(
            approximate, self.params.column_x, self.params.column_y
        )
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

    def create_dependent_modules(
        self,
        input_module: Module,
        input_slot: str,
        min_: Optional[Module] = None,
        max_: Optional[Module] = None,
        min_value: Union[None, bool, Module] = None,
        max_value: Union[None, bool, Module] = None,
        **kwds: Any,
    ) -> RangeQuery2D:
        """
        Creates a default configuration containing the necessary underlying modules.
        Beware, {min,max}_value=None is not the same as {min,max}_value=False.
        With None, a min module is created and connected.
        With False, it is not created and not connected.
        """
        from progressivis import Variable

        if self.input_module is not None:  # test if already called
            return self
        with self.grouped():
            scheduler = self.scheduler
            params = self.params
            self.input_module = input_module
            self.input_slot = input_slot
            with scheduler:
                index_2d = BinningIndexND(
                    group=self.name,
                    scheduler=scheduler,
                )
                self.dep.index = index_2d
                index_2d.input.table = input_module.output[input_slot][
                    params.column_x,
                    params.column_y,
                ]
                if min_value is None:
                    min_value = Variable(group=self.name, scheduler=scheduler)
                if max_value is None:
                    max_value = Variable(group=self.name, scheduler=scheduler)
                range_query = self
                range_query.input.index = index_2d.output.result
                range_query.input.timestamps_x = index_2d.output.bin_timestamps_0
                range_query.input.timestamps_y = index_2d.output.bin_timestamps_1
                if min_value:
                    assert isinstance(min_value, Module)
                    range_query.input.lower = min_value.output.result
                if max_value:
                    assert isinstance(max_value, Module)
                    range_query.input.upper = max_value.output.result
                range_query.input.min = (
                    index_2d.output.min_out if min_ is None else min_.output.result
                )
                range_query.input.max = (
                    index_2d.output.max_out if max_ is None else max_.output.result
                )
            self.dep.min = min_
            self.dep.max = max_
            self.dep.min_value = min_value
            self.dep.max_value = max_value
            return range_query

    def _create_min_max(self) -> None:
        if self._min_table is None:
            self._min_table = PDict({self._column_x: np.inf, self._column_y: np.inf})
        if self._max_table is None:
            self._max_table = PDict({self._column_x: -np.inf, self._column_y: -np.inf})

    def _set_minmax_out(self, attr_: str, val_x: float, val_y: float) -> None:
        d = {self._column_x: val_x, self._column_y: val_y}
        if getattr(self, attr_) is None:
            setattr(self, attr_, PDict(d))
        else:
            getattr(self, attr_).update(d)

    def _set_min_out(self, val_x: float, val_y: float) -> None:
        return self._set_minmax_out("_min_table", val_x, val_y)

    def _set_max_out(self, val_x: float, val_y: float) -> None:
        return self._set_minmax_out("_max_table", val_x, val_y)

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        """ """
        index_2d_slot = self.get_input_slot("index")
        if index_2d_slot.data() is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        input_slot = index_2d_slot
        # input_slot.update(run_number)
        # X-Y common func

        def _ts_func(
            tstamps: Slot, ts_data: dict[Any, Any], ts_changes: PIntSet
        ) -> PIntSet:
            ts_k_ids = {ts_data.k_(i): i for i in ts_changes}  # type: ignore
            if -1 in ts_k_ids and ts_k_ids[-1] in tstamps.updated.changes:
                tstamps.reset()
                # print("Tstamp reset")
                return PIntSet()
            else:
                ts_k_ids.pop(-1, None)  # removing -1 key if present (at creation)
                return PIntSet(ts_k_ids.values())  # relevant bins

        # timestamps on X
        only_bins_x = PIntSet()
        tstamps_x = tstamps_y = None
        if self.has_input_slot("timestamps_x"):
            tstamps_x = self.get_input_slot("timestamps_x")
            ts_data_x = tstamps_x.data()
            ts_changes_x = tstamps_x.created.changes | tstamps_x.updated.changes
            only_bins_x = _ts_func(tstamps_x, ts_data_x, ts_changes_x)
        only_bins_y = PIntSet()
        if self.has_input_slot("timestamps_y"):
            tstamps_y = self.get_input_slot("timestamps_y")
            ts_data_y = tstamps_y.data()
            ts_changes_y = tstamps_y.created.changes | tstamps_y.updated.changes
            only_bins_y = _ts_func(tstamps_y, ts_data_y, ts_changes_y)
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
        upper_slot = self.get_input_slot("upper")
        limit_changed = False
        if (
            lower_slot.updated.any()
            or lower_slot.created.any()
            or upper_slot.updated.any()
            or upper_slot.created.any()
        ):
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
        # X ...
        lower_value_x = lower_slot.data().get(self._watched_key_lower_x)
        upper_value_x = upper_slot.data().get(self._watched_key_upper_x)
        # Y ...
        lower_value_y = lower_slot.data().get(self._watched_key_lower_y)
        upper_value_y = upper_slot.data().get(self._watched_key_upper_y)
        # X ...
        minv_x = min_slot.data().get(self._column_x)
        maxv_x = max_slot.data().get(self._column_x)
        # Y ...
        minv_y = min_slot.data().get(self._column_y)
        maxv_y = max_slot.data().get(self._column_y)
        # X ...
        if (
            lower_value_x is None
            or np.isnan(lower_value_x)
            or np.isinf(lower_value_x)
            or lower_value_x < minv_x
            or lower_value_x >= maxv_x
        ):
            lower_value_x = -float("inf")
        if (
            upper_value_x is None
            or np.isnan(upper_value_x)
            or upper_value_x > maxv_x
            or upper_value_x <= minv_x
            or upper_value_x <= lower_value_x
        ):
            upper_value_x = float("inf")
        # Y ...
        if (
            lower_value_y is None
            or np.isnan(lower_value_y)
            or lower_value_y < minv_y
            or lower_value_y >= maxv_y
        ):
            lower_value_y = -float("inf")
        if (
            upper_value_y is None
            or np.isnan(upper_value_y)
            or upper_value_y > maxv_y
            or upper_value_y <= minv_y
            or upper_value_y <= lower_value_y
        ):
            upper_value_y = float("inf")
        self._set_min_out(
            minv_x if np.isinf(lower_value_x) else lower_value_x,
            minv_y if np.isinf(lower_value_y) else lower_value_y,
        )
        self._set_max_out(
            maxv_x if np.isinf(upper_value_x) else upper_value_x,
            maxv_y if np.isinf(upper_value_y) else upper_value_y,
        )
        if steps == 0 and not limit_changed:
            return self._return_run_step(self.state_blocked, steps_run=0)
        # ...

        assert self._impl
        if not self._impl.is_started:
            self._impl.start(
                input_table,
                cast(BinningIndexND, index_2d_slot.output_module),
                lower_value_x,
                upper_value_x,
                lower_value_y,
                upper_value_y,
                limit_changed,
                created=created,
                updated=updated,
                deleted=deleted,
                only_bins_x=only_bins_x,
                only_bins_y=only_bins_y,
            )
        else:
            self._impl.resume(
                cast(BinningIndexND, index_2d_slot.output_module),
                lower_value_x,
                upper_value_x,
                lower_value_y,
                upper_value_y,
                limit_changed,
                created=created,
                updated=updated,
                deleted=deleted,
                only_bins_x=only_bins_x,
                only_bins_y=only_bins_y,
            )
        if not input_slot.has_buffered():
            if tstamps_x is not None:
                tstamps_x.clear_buffers()
            if tstamps_y is not None:
                tstamps_y.clear_buffers()
        assert self._impl.result
        self.result.selection = self._impl.result._values
        return self._return_run_step(self.next_state(input_slot), steps)
