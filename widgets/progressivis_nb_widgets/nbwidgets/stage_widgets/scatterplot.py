from .utils import (
    make_button,
    stage_register,
    VBoxSchema,
)
from ..utils import historized_widget
from ._multi_series import scatterplot_no_data
import ipywidgets as ipw
from vega.widget import VegaWidget
import pandas as pd
from progressivis.core import Scheduler
from typing import Any as AnyType, Dict, cast, Type, List
from typing_extensions import TypeAlias
import copy

WidgetType = AnyType
_l = ipw.Label

N = 4  # 1X + 3Y


_VegaWidget: TypeAlias = cast(Type[AnyType], historized_widget(VegaWidget, "update"))


class ScatterplotW(VBoxSchema):
    def init(self) -> None:
        self.output_dtypes = None
        self._axis = {}
        lst: List[ipw.DOMWidget] = [_l("Axis"), _l("PColumn"),  _l("Symbol")]
        for row_name in ["X", "Y", "Color", "Shape"]:
            row = self._axis_row(row_name)
            self._axis[row_name] = row
            lst.extend(row.values())
        gb = ipw.GridBox(
            lst,
            layout=ipw.Layout(grid_template_columns="5% 40% 40%"),
        )
        btn_apply = self._btn_ok = make_button(
            "Apply", disabled=True, cb=self._btn_apply_cb
        )
        self.set_schema(dict(grid=gb, btn_apply=btn_apply, vega=None))

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
        sym = ipw.Text(
            value="", placeholder="optional", description="", disabled=False
        )
        return dict(axis=axis_w, col=col, sym=sym)

    def _update_vw(self, s: Scheduler, run_number: int) -> None:
        assert hasattr(self.input_module, "result")
        tbl = self.input_module.result
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
        cast(_VegaWidget, self["vega"]).update("data", remove="true", insert=df)

    def _col_xy_cb(self, change: Dict[str, AnyType]) -> None:
        self._x_col = self._axis["X"]["col"].value
        self._y_col = self._axis["Y"]["col"].value
        self["btn_apply"].disabled = not (self._x_col and self._y_col)

    def _btn_apply_cb(self, btn: AnyType) -> None:
        sc_json: Dict[str, AnyType] = copy.deepcopy(scatterplot_no_data)
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
        self["vega"] = _VegaWidget(spec=sc_json)
        self.input_module.scheduler().on_tick(self._update_vw)
        self.dag_running()


stage_register["Scatterplot"] = ScatterplotW
