from .utils import make_button, stage_register, VBoxSchema, SchemaBase
from ..utils import historized_widget
from ._multi_series import histogram1d_no_data
import ipywidgets as ipw
from vega.widget import VegaWidget
import pandas as pd
from progressivis.core import Scheduler

from typing import Any as AnyType, Type, cast, List, Dict, Callable
from typing_extensions import TypeAlias
import copy


_l = ipw.Label

N = 4  # 1X + 3Y


HVegaWidget: TypeAlias = cast(Type[AnyType],
                              historized_widget(VegaWidget, "update"))  # noqa: F821


class HistogramW(VBoxSchema):
    class Schema(SchemaBase):
        grid: ipw.GridBox
        btn_apply: ipw.Button
        vega: HVegaWidget

    child: Schema

    def init(self) -> None:
        self.output_dtypes = None
        self._axis = {}
        lst: List[ipw.DOMWidget] = [
            _l("Axis"),
            _l("PColumn"),
            _l("Symbol"),
            _l("Aggregate"),
        ]
        for row_name in ["X", "Y"]:
            row = self._axis_row(row_name)
            self._axis[row_name] = row
            lst.extend(row.values())
        self.child.grid = ipw.GridBox(
            lst,
            layout=ipw.Layout(grid_template_columns="5% 30% 30% 20%"),
        )
        self.child.btn_apply = self._btn_ok = make_button(
            "Apply", disabled=True, cb=self._btn_apply_cb
        )

    def _axis_row(self, axis: str) -> Dict[str, ipw.DOMWidget]:
        axis_w = _l(axis)
        col_list = [""] + list(self.dtypes.keys())
        col = ipw.Dropdown(
            options=col_list,
            description="",
            disabled=False,
            layout={"width": "initial"},
        )
        col.observe(self._col_xy_cb, "value")
        sym = ipw.Text(value="", placeholder="optional", description="", disabled=False)
        aggregate = ipw.Dropdown(
            options=["", "sum", "mean"],
            description="",
            disabled=False,
            layout={"width": "initial"},
        )
        aggregate.observe(self._make_aggregate_cb(axis), "value")
        return dict(axis=axis_w, col=col, sym=sym, aggregate=aggregate)

    def _update_vw(self, s: Scheduler, run_number: int) -> None:
        assert hasattr(self.input_module, "result")
        tbl = self.input_module.result
        if tbl is None:
            print("no tbl")
            return
        x_arr = tbl[self._x_col].loc[:]
        y_arr = tbl[self._y_col].loc[:]
        df_dict = {self._x_sym: x_arr, self._y_sym: y_arr}
        df = pd.DataFrame(df_dict)
        self.child.vega.update("data", remove="true", insert=df)

    def _make_aggregate_cb(self, axis: str) -> Callable[..., None]:
        other = "Y" if axis == "X" else "X"

        def _aggregate_cb(change: Dict[str, AnyType]) -> None:
            if not change["new"]:
                return
            self._axis[other]["aggregate"].value = ""

        return _aggregate_cb

    def _col_xy_cb(self, change: Dict[str, AnyType]) -> None:
        self._x_col = self._axis["X"]["col"].value
        self._y_col = self._axis["Y"]["col"].value
        self.child.btn_apply.disabled = not (self._x_col and self._y_col)

    def _btn_apply_cb(self, btn: AnyType) -> None:
        sc_json: Dict[str, AnyType] = copy.deepcopy(histogram1d_no_data)
        x_sym = self._axis["X"]["sym"].value
        x_aggr = self._axis["X"]["aggregate"].value
        if x_aggr:
            x_kw = {"aggregate": x_aggr}
        else:
            x_kw = {"bin": True}
        y_sym = self._axis["Y"]["sym"].value
        y_aggr = self._axis["Y"]["aggregate"].value
        if y_aggr:
            y_kw = {"aggregate": y_aggr}
        else:
            y_kw = {"bin": True}
        self._x_sym = x_sym or self._x_col
        self._y_sym = y_sym or self._y_col
        sc_json["encoding"]["x"] = {"field": self._x_sym, **x_kw}
        sc_json["encoding"]["y"] = {"field": self._y_sym, **y_kw}
        self.child.vega = HVegaWidget(spec=sc_json)
        self.input_module.scheduler().on_tick(self._update_vw)
        self.dag_running()


stage_register["Histogram1D"] = HistogramW
