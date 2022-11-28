from .utils import (
    make_button,
    stage_register,
    dongle_widget,
    set_child,
    VBox,
)
from ._multi_series import scatterplot_no_data
from .. import PrevImages
import ipywidgets as ipw  # type: ignore
import time
from vega.widget import VegaWidget  # type: ignore
import pandas as pd
from typing import Any as AnyType, Dict
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


class ScatterplotW(VBox):
    def __init__(self) -> None:
        super().__init__()

    def init(self):
        self.output_dtypes = None
        self._axis = {}
        lst = [_l("Axis"), _l("Column"),  _l("Symbol")]
        for row_name in ["X", "Y", "Color", "Shape"]:
            row = self._axis_row(row_name)
            self._axis[row_name] = row
            lst.extend(row.values())
        self._gb = ipw.GridBox(
            lst,
            layout=ipw.Layout(grid_template_columns="5% 40% 40%"),
        )
        self._btn_apply = self._btn_ok = make_button(
            "Apply", disabled=True, cb=self._btn_apply_cb
        )
        self._vw = dongle_widget()
        self.children = (self._gb, self._btn_apply, self._vw)

    def _axis_row(self, axis):
        axis_w = _l(axis)
        col_list = [""] + list(self.dtypes.keys())
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
        return dict(axis=axis_w, col=col, sym=sym)

    def _update_vw(self, m, run_number):
        tbl = self.input_module.table
        if tbl is None:
            print("no tbl")
            return
        x_arr = tbl[self._x_col].loc[:]
        y_arr = tbl[self._y_col].loc[:]
        df_dict = {self._x_sym: x_arr, self._y_sym: y_arr}
        if self._color_col:
            color_arr = tbl[self._color_col].loc[:]
            df_dict[self._color_sym] = color_arr
        if self._shape_col:
            shape_arr = tbl[self._shape_col].loc[:]
            df_dict[self._shape_sym] = shape_arr
        df = pd.DataFrame(df_dict)
        self._vw.update("data", remove="true", insert=df)

    def _col_xy_cb(self, change: Dict[str, AnyType]) -> None:
        self._x_col = self._axis["X"]["col"].value
        self._y_col = self._axis["Y"]["col"].value
        if self._x_col and self._y_col:
            self._btn_apply.disabled = False
        else:
            self._btn_apply.disabled = True

    def _btn_apply_cb(self, btn):
        sc_json = copy.deepcopy(scatterplot_no_data)
        x_sym = self._axis["X"]["sym"].value
        y_sym = self._axis["Y"]["sym"].value
        self._x_sym = x_sym or self._x_col
        self._y_sym = y_sym or self._y_col
        sc_json["encoding"]["x"] = {"field": self._x_sym, "type": "quantitative"}
        sc_json["encoding"]["y"] = {"field": self._y_sym, "type": "quantitative"}
        self._color_col = self._axis["Color"]["col"].value
        if self._color_col:
            color_sym = self._axis["Color"]["sym"].value
            self._color_sym = color_sym or self._color_col
            sc_json["encoding"]["color"] = {"field": self._color_sym, "type": "nominal"}
        self._shape_col = self._axis["Shape"]["col"].value
        if self._shape_col:
            shape_sym = self._axis["Shape"]["sym"].value
            self._shape_sym = shape_sym or self._shape_col
            sc_json["encoding"]["shape"] = {"field": self._shape_sym, "type": "nominal"}
        self._vw = _VegaWidget(spec=sc_json)
        set_child(self, 2, self._vw)
        self.input_module.scheduler().on_tick(self._update_vw)
        self.dag_running()


stage_register["Scatterplot"] = ScatterplotW
