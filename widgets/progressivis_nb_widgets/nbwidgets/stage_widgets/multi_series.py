from .utils import (
    make_button,
    stage_register,
    dongle_widget,
    set_child,
    VBox,
)
from ._multi_series import multi_series_no_data
from .. import PrevImages
import ipywidgets as ipw  # type: ignore
import time
from vega.widget import VegaWidget  # type: ignore
import pandas as pd
from typing import Any as AnyType, Dict

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


class MultiSeriesW(VBox):
    def __init__(self) -> None:
        super().__init__()

    def init(self):
        self.output_dtypes = None
        self._axis = []
        lst = [_l("Axis"), _l("Column"), _l("* Factor"), _l("Symbol")]
        for i in range(N):
            row = self._axis_row("Y" if i else "X")
            self._axis.append(row)
            lst.extend(row.values())
        self._gb = ipw.GridBox(
            lst,
            layout=ipw.Layout(grid_template_columns="5% 40% 20% 20%"),
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
        if axis.upper() == "Y":
            factor = ipw.FloatText(value=1.0, description="", disabled=False)
            sym = ipw.Text(
                value="", placeholder="optional", description="", disabled=False
            )
        else:  # i.e. "X"
            factor = dongle_widget()
            sym = dongle_widget()
        return dict(axis=axis_w, col=col, factor=factor, sym=sym)

    def _update_vw(self, m, run_number):
        tbl = self.input_module.table
        if tbl is None:
            print("no tbl")
            return
        x_col = self._axis[0]["col"].value
        dataframes = []
        for y_row in self._axis[1:]:
            y_col = y_row["col"].value
            if not y_col:
                continue
            factor = y_row["factor"].value
            sym_v = y_row["sym"].value or y_col
            if factor != 1:
                sym_v = f"{sym_v}*{factor}"
            df = pd.DataFrame(
                {
                    "date": [
                        f"{y}-{m}-{d}" for (y, m, d, _, _, _) in tbl[x_col].loc[:]
                    ],
                    "level": tbl[y_col].loc[:] * factor,
                    "symbol": [sym_v] * len(tbl),
                }
            )
            dataframes.append(df)
        self._vw.update("data", remove="true", insert=pd.concat(dataframes))

    def _col_xy_cb(self, change: Dict[str, AnyType]) -> None:
        has_x = False
        has_y = False
        for row in self._axis:
            if row["col"].value:
                if row["axis"].value == "X":
                    has_x = True
                else:
                    has_y = True
        if has_x and has_y:
            self._btn_apply.disabled = False
        else:
            self._btn_apply.disabled = True

    def _btn_apply_cb(self, btn):
        self._vw = _VegaWidget(spec=multi_series_no_data)
        set_child(self, 2, self._vw)
        self.input_module.on_after_run(self._update_vw)
        # self._input_module.scheduler().on_tick(self._update_vw)
        self.dag_running()


stage_register["MultiSeries"] = MultiSeriesW
