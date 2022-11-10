from __future__ import annotations
import logging
from itertools import product
from ..core import Sink
from ..stats import (
    Min,
    Max,
    KLLSketch,
    Histogram1D,
    Histogram2D,
    Histogram1DCategorical,
    Var,
    Distinct,
    Corr,
)
from ..core.module import GroupContext
from ..table.module import TableModule
from ..table.table import Table
from ..table.pattern import Pattern
from ..core.slot import SlotDescriptor
from ..core.decorators import process_slot, run_if_any
from ..table.dshape import dshape_fields
from ..table.range_query import RangeQuery
from ..utils.psdict import PsDict
from ..io import DynVar
from typing import Any, Dict, Callable, Optional, TYPE_CHECKING, cast
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..core.module import ReturnRunStep


logger = logging.getLogger(__name__)


class Histogram1dPattern(Pattern):
    def __init__(self, column: str, factory: StatsFactory, **kwds: Any) -> None:
        """ """
        super().__init__(**kwds)
        self._column = column
        self._factory = factory

    def create_dependent_modules(self) -> None:
        super().create_dependent_modules()
        input_module = self._factory._input_module
        input_slot = self._factory._input_slot
        scheduler = self._factory.scheduler()
        col = self._column
        with scheduler:
            # TODO replace sink with a real dependency
            self.kll = KLLSketch(column=col, scheduler=scheduler)
            self.kll.params.binning = 128
            self.kll.input.table = input_module.output[input_slot]
            assert self.sink
            self.sink.input.inp = self.kll.output.result
            # TODO: reuse min max
            self.max = Max(scheduler=scheduler, columns=[col])
            self.max.input.table = input_module.output[input_slot]
            self.min = Min(scheduler=scheduler, columns=[col])
            self.min.input.table = input_module.output[input_slot]
            self.lower = DynVar({col: "*"}, scheduler=scheduler)
            self.upper = DynVar({col: "*"}, scheduler=scheduler)
            self.range_query = RangeQuery(
                scheduler=scheduler, column=col, columns=[col], approximate=True
            )
            self.range_query.params.column = col
            self.range_query.create_dependent_modules(
                input_module,
                input_slot,
                min_=self.min,
                max_=self.max,
                min_value=self.lower,
                max_value=self.upper,
            )
            self.histogram1d = Histogram1D(scheduler=scheduler, column=col)
            self.histogram1d.input.table = self.range_query.output.result
            self.histogram1d.input.min = self.range_query.output.min
            self.histogram1d.input.max = self.range_query.output.max
            sink = Sink(scheduler=scheduler)
            sink.input.inp = self.histogram1d.output.result


class Histogram2dPattern(Pattern):
    def __init__(
        self, x_column: str, y_column: str, factory: StatsFactory, **kwds: Any
    ) -> None:
        """ """
        super().__init__(**kwds)
        self._x_column = x_column
        self._y_column = y_column
        self._factory = factory

    def create_dependent_modules(self) -> None:
        super().create_dependent_modules()
        input_module = self._factory._input_module
        input_slot = self._factory._input_slot
        scheduler = self._factory.scheduler()
        x_col, y_col = self._x_column, self._y_column
        with scheduler:
            # TODO: reuse min max
            self.max = Max(scheduler=scheduler, columns=[x_col, y_col])
            self.max.input.table = input_module.output[input_slot]
            self.min = Min(scheduler=scheduler, columns=[x_col, y_col])
            self.min.input.table = input_module.output[input_slot]
            self.histogram2d = Histogram2D(
                x_column=x_col, y_column=y_col, scheduler=scheduler
            )
            self.histogram2d.input.table = input_module.output[input_slot]
            self.histogram2d.input.min = self.min.output.result
            self.histogram2d.input.max = self.max.output.result
            self.histogram2d.params.xbins = 64
            self.histogram2d.params.ybins = 64
            sink = Sink(scheduler=scheduler)
            sink.input.inp = self.histogram2d.output.result


class DataShape(TableModule):
    """
    Adds statistics on input data
    """

    inputs = [
        SlotDescriptor("table", type=Table, required=True),
    ]

    def __init__(self, **kwds: Any) -> None:
        """ """
        super().__init__(**kwds)
        pass

    def reset(self):
        pass

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            slot = ctx.table
            data = slot.data()
            if not data:
                return self._return_run_step(self.state_blocked, steps_run=0)
            if slot.has_buffered():
                slot.clear_buffers()
            self.result = PsDict({k: str(v) for (k, v) in dshape_fields(data.dshape)})
            return self._return_run_step(self.state_zombie, steps_run=0)


def _add_max_col(col: str, factory: StatsFactory) -> Max:
    input_module = factory._input_module
    scheduler = factory.scheduler()
    with scheduler:
        m = Max(columns=[col], scheduler=scheduler)
        m.input.table = input_module.output.result
        sink = Sink(scheduler=scheduler)
        sink.input.inp = m.output.result
        return m


def _add_min_col(col: str, factory: StatsFactory) -> Min:
    input_module = factory._input_module
    scheduler = factory.scheduler()
    with scheduler:
        m = Min(columns=[col], scheduler=scheduler)
        m.input.table = input_module.output.result
        sink = Sink(scheduler=scheduler)
        sink.input.inp = m.output.result
        return m


def _add_var_col(col: str, factory: StatsFactory) -> Var:
    input_module = factory._input_module
    scheduler = factory.scheduler()
    with scheduler:
        m = Var(columns=[col], scheduler=scheduler)
        m.input.table = input_module.output.result
        sink = Sink(scheduler=scheduler)
        sink.input.inp = m.output.result
        return m


def _add_distinct_col(col: str, factory: StatsFactory) -> Distinct:
    input_module = factory._input_module
    scheduler = factory.scheduler()
    with scheduler:
        m = Distinct(columns=[col], scheduler=scheduler)
        m.input.table = input_module.output.result
        sink = Sink(scheduler=scheduler)
        sink.input.inp = m.output.result
        return m


def _add_correlation(col: str, factory: StatsFactory) -> Optional[Corr]:
    _ = col  # keeps mypy happy ...
    df = factory.last_selection.get("matrix")
    columns = df.index[df.loc[:, "corr"]]
    columns = list(columns)
    prev_corr: Optional[Corr] = cast(
        Optional[Corr], factory._multi_col_modules.get("corr")
    )
    if prev_corr is not None:
        if prev_corr._columns == columns:
            return prev_corr
        # list of column changed => remove old module
        factory._to_delete.append(prev_corr.name)
        del factory._multi_col_modules["corr"]
    if len(columns) < 2:
        return None
    input_module = factory._input_module
    scheduler = factory.scheduler()
    with scheduler:
        m = Corr(columns=list(columns), scheduler=scheduler)
        m.input.table = input_module.output.result
        sink = Sink(scheduler=scheduler)
        sink.input.inp = m.output.result
        factory._multi_col_modules["corr"] = m
        return m


def _add_barplot_col(col: str, factory: StatsFactory) -> Histogram1DCategorical:
    input_module = factory._input_module
    scheduler = factory.scheduler()
    with scheduler:
        m = Histogram1DCategorical(scheduler=scheduler, column=col)
        m.input.table = input_module.output.result
        sink = Sink(scheduler=scheduler)
        sink.input.inp = m.output.result
        return m


def _add_hist_col(col: str, factory: StatsFactory) -> TableModule:
    assert factory.types
    col_type = factory.types[col]
    if col_type == "string":
        return _add_barplot_col(col, factory)
    scheduler = factory.scheduler()
    with scheduler:
        m = Histogram1dPattern(column=col, factory=factory, scheduler=scheduler)
        # sink = Sink(scheduler=scheduler)
        # sink.input.inp = m.output.result
        m.create_dependent_modules()
        return m


def _pass_func(col: str, factory: StatsFactory) -> None:
    print("Pass func")


def _hide_func(col: str, factory: StatsFactory) -> None:
    raise ValueError("hide function should never be called ...")


def _h2d_func(cx: str, cy: str, factory: StatsFactory) -> Histogram2dPattern:
    scheduler = factory.scheduler()
    with scheduler:
        m = Histogram2dPattern(
            x_column=cx, y_column=cy, factory=factory, scheduler=scheduler
        )
        # sink = Sink(scheduler=scheduler)
        # sink.input.inp = m.output.result
        m.create_dependent_modules()
        return m


class StatsFactory(TableModule):
    """
    Adds statistics on input data
    """

    inputs = [
        SlotDescriptor("table", type=Table, required=True),
        SlotDescriptor("selection", type=PsDict, required=True),
    ]

    def __init__(
        self, input_module: TableModule, input_slot: str = "result", **kwds: Any
    ) -> None:
        """ """
        super().__init__(**kwds)
        self._input_module = input_module
        self._input_slot = input_slot
        self._matrix: Optional[pd.DataFrame] = None
        self._h2d_matrix: Optional[pd.DataFrame] = None
        self.types: Optional[Dict[str, str]] = None
        self._multi_col_funcs = set(["corr"])
        self._multi_col_modules: Dict[str, Optional[TableModule]] = {}
        self.func_dict: Dict[str, Callable] = dict(
            hide=_hide_func,
            max=_add_max_col,
            min=_add_min_col,
            var=_add_var_col,
            hist=_add_hist_col,
            distinct=_add_distinct_col,
            corr=_add_correlation,
        )
        self._sink = None

    def reset(self):
        pass

    @process_slot("table", "selection", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with GroupContext(self), self.context as ctx:
            slot = ctx.table
            data = slot.data()
            if not data:
                return self._return_run_step(self.state_blocked, steps_run=0)
            if slot.has_buffered():
                slot.clear_buffers()
            if self._matrix is None:
                self.types = {k: str(v) for (k, v) in dshape_fields(data.dshape)}
                cols = [k for (k, v) in dshape_fields(data.dshape)]
                funcs = self.func_dict.keys()
                arr = np.zeros((len(cols), len(funcs)), dtype=object)
                self._matrix = pd.DataFrame(arr, index=cols, columns=funcs)
            if self._h2d_matrix is None:
                assert self.types
                num_cols = [
                    k
                    for (k, _) in dshape_fields(data.dshape)
                    if self.types[k] != "string"
                ]
                len_ = len(num_cols)
                arr = np.zeros((len_, len_), dtype=object)
                self._h2d_matrix = pd.DataFrame(arr, index=num_cols, columns=num_cols)
            sel_slot = ctx.selection
            if not sel_slot.has_buffered():
                return self._return_run_step(self.state_blocked, steps_run=0)
            sel_slot.clear_buffers()
            self.last_selection = sel_slot.data()
            hidden_cols = self.last_selection.get("hidden_cols", [])
            self._to_delete = []
            for col in hidden_cols:
                for cell in self._matrix.loc[col, :]:
                    if cell:
                        self._to_delete.append(cell.name)
                self._matrix.loc[col, :] = 0
            for col in hidden_cols:
                for cell in self._h2d_matrix.loc[col, :]:
                    if cell:
                        self._to_delete.append(cell.name)
                self._h2d_matrix.loc[col, :] = 0
            for col in hidden_cols:
                for cell in self._h2d_matrix.loc[:, col]:
                    if cell:
                        self._to_delete.append(cell.name)
                self._h2d_matrix.loc[col, :] = 0
            scheduler = self.scheduler()
            steps = len(self._to_delete) // 2  # or more ?
            df = self.last_selection.get("matrix")
            if df is not None:
                for func in df.columns:
                    if func in self._multi_col_funcs:
                        self._multi_col_modules[func] = self.func_dict[func]("", self)
                        steps += 1
                        continue
                    for attr in df.index:
                        cell = df.loc[attr, func]
                        if cell:
                            if not self._matrix.loc[attr, func]:
                                self._matrix.loc[attr, func] = self.func_dict[func](
                                    attr, self
                                )
                        else:
                            if self._matrix.loc[attr, func]:
                                self._to_delete.append(
                                    self._matrix.loc[attr, func].name
                                )
                                self._matrix.loc[attr, func] = 0
                        steps += 1
            df = self.last_selection.get("h2d_matrix")
            if df is not None:
                for cx, cy in product(df.columns, repeat=2):
                    if cx == cy:
                        continue
                    cell = df.loc[cx, cy]
                    if cell:
                        if not self._h2d_matrix.loc[cx, cy]:
                            self._h2d_matrix.loc[cx, cy] = _h2d_func(cx, cy, self)
                    else:
                        if self._h2d_matrix.loc[cx, cy]:
                            self._to_delete.append(self._h2d_matrix.loc[cx, cy].name)
                            self._h2d_matrix.loc[cx, cy] = 0
                        steps += 1
            if self._to_delete:
                with scheduler as dataflow:
                    deps = dataflow.collateral_damage(*self._to_delete)
                    dataflow.delete_modules(*deps)
                    # pass
            return self._return_run_step(self.state_blocked, steps_run=steps)

    def create_dependent_modules(self, var_name=None):
        s = self.scheduler()
        self.variable = DynVar(name=var_name, scheduler=s)
        self.input.selection = self.variable.output.result
