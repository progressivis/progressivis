from .utils import (
    make_button,
    stage_register,
    dongle_widget,
    set_child,
    ChainingWidget,
)
from ._multi_series import histogram1d_no_data
from .. import PrevImages
import ipywidgets as ipw  # type: ignore
import time
from vega.widget import VegaWidget  # type: ignore
import pandas as pd
from progressivis.table.module import TableModule
from typing import Any as AnyType, Dict, Callable
import copy

WidgetType = AnyType
_l = ipw.Label

N = 4  # 1X + 3Y


class VegaWidgetHz(ipw.VBox):
    def __init__(self, *args, **kw):
        self.vega_wg = VegaWidget(*args, **kw)
        self.classname = f"vegawidget-{id(self.vega_wg)}"
        self.vega_wg.add_class(self.classname)
        self.pim = PrevImages()
        self.pim.target = self.classname
        super().__init__([self.vega_wg, self.pim])

    def update(self, *args, **kw):
        self.vega_wg.update(*args, **kw)
        time.sleep(0.1)
        self.pim.update()


_VegaWidget = VegaWidgetHz


class HistogramW(ipw.VBox, ChainingWidget):
    def __init__(
        self,
        parent: AnyType,
        dtypes: Dict[str, AnyType],
        input_module: TableModule,
        input_slot: str = "result",
        dag=None,
    ) -> None:
        super().__init__(
            parent=parent,
            dtypes=dtypes,
            input_module=input_module,
            input_slot=input_slot,
            dag=dag,
        )
        self.dag_register()
        self._output_dtypes = None
        self._axis = {}
        lst = [_l("Axis"), _l("Column"),  _l("Symbol"), _l("Aggregate")]
        for row_name in ["X", "Y"]:
            row = self._axis_row(row_name)
            self._axis[row_name] = row
            lst.extend(row.values())
        self._gb = ipw.GridBox(
            lst,
            layout=ipw.Layout(grid_template_columns="5% 30% 30% 20%"),
        )
        self._btn_apply = self._btn_ok = make_button(
            "Apply", disabled=True, cb=self._btn_apply_cb
        )
        self._vw = dongle_widget()
        self.children = (self._gb, self._btn_apply, self._vw)

    def _axis_row(self, axis):
        axis_w = _l(axis)
        col_list = [""] + list(self._dtypes.keys())
        col = ipw.Dropdown(
            options=col_list,
            description="",
            disabled=False,
            layout={"width": "initial"},
        )
        col.observe(self._col_xy_cb, "value")
        sym = ipw.Text(
            value="", placeholder="optional", description="", disabled=False
        )
        aggregate = ipw.Dropdown(
            options=["", "sum", "mean"],
            description="",
            disabled=False,
            layout={"width": "initial"},
        )
        aggregate.observe(self._make_aggregate_cb(axis), "value")
        return dict(axis=axis_w, col=col, sym=sym, aggregate=aggregate)

    def _update_vw(self, m, run_number):
        tbl = self._input_module.table
        if tbl is None:
            print("no tbl")
            return
        x_arr = tbl[self._x_col].loc[:]
        y_arr = tbl[self._y_col].loc[:]
        df_dict = {self._x_sym: x_arr, self._y_sym: y_arr}
        df = pd.DataFrame(df_dict)
        self._vw.update("data", remove="true", insert=df)

    def _make_aggregate_cb(self, axis) -> Callable:
        other = "Y" if axis == "X" else "X"

        def _aggregate_cb(change: Dict[str, AnyType]) -> None:
            if not change["new"]:
                return
            self._axis[other]["aggregate"].value = ""
        return _aggregate_cb

    def _col_xy_cb(self, change: Dict[str, AnyType]) -> None:
        self._x_col = self._axis["X"]["col"].value
        self._y_col = self._axis["Y"]["col"].value
        if self._x_col and self._y_col:
            self._btn_apply.disabled = False
        else:
            self._btn_apply.disabled = True

    def _btn_apply_cb(self, btn):
        sc_json = copy.deepcopy(histogram1d_no_data)
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
        self._vw = _VegaWidget(spec=sc_json)
        set_child(self, 2, self._vw)
        self._input_module.scheduler().on_tick(self._update_vw)
        self.dag_running()


stage_register["Histogram1D"] = HistogramW
