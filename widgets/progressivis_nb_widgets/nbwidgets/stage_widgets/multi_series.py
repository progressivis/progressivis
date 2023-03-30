from .utils import make_button, stage_register, dongle_widget, VBoxSchema, SchemaBase
from ..utils import historized_widget
from ._multi_series import multi_series_no_data
import ipywidgets as ipw
from vega.widget import VegaWidget
import pandas as pd
from progressivis.core import Module
from typing import Any as AnyType, Dict, cast, Type, List, Union
from typing_extensions import TypeAlias


_l = ipw.Label

N = 4  # 1X + 3Y


HVegaWidget: TypeAlias = cast(Type[AnyType],
                              historized_widget(VegaWidget, "update"))  # noqa: F821


class MultiSeriesW(VBoxSchema):
    class Schema(SchemaBase):
        grid: ipw.GridBox
        btn_apply: ipw.Button
        vega: HVegaWidget

    def init(self) -> None:
        self.output_dtypes = None
        self._axis = []
        lst: List[ipw.DOMWidget] = [
            _l("Axis"),
            _l("PColumn"),
            _l("* Factor"),
            _l("Symbol"),
        ]
        for i in range(N):
            row = self._axis_row("Y" if i else "X")
            self._axis.append(row)
            lst.extend(row.values())
        self.child.grid = ipw.GridBox(
            lst,
            layout=ipw.Layout(grid_template_columns="5% 40% 20% 20%"),
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
        factor: Union[ipw.FloatText, ipw.Label]
        sym: Union[ipw.Text, ipw.Label]
        if axis.upper() == "Y":
            factor = ipw.FloatText(value=1.0, description="", disabled=False)
            sym = ipw.Text(
                value="", placeholder="optional", description="", disabled=False
            )
        else:  # i.e. "X"
            factor = dongle_widget()
            sym = dongle_widget()
        return dict(axis=axis_w, col=col, factor=factor, sym=sym)

    def _update_vw(self, m: Module, run_number: int) -> None:
        assert hasattr(self.input_module, "result")
        tbl = self.input_module.result
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
        self.child.vega.update("data", remove="true", insert=pd.concat(dataframes))

    def _col_xy_cb(self, change: Dict[str, AnyType]) -> None:
        has_x = False
        has_y = False
        for row in self._axis:
            if row["col"].value:
                if row["axis"].value == "X":
                    has_x = True
                else:
                    has_y = True
        self.child.btn_apply.disabled = not (has_x and has_y)

    def _btn_apply_cb(self, btn: AnyType) -> None:
        self.child.vega = HVegaWidget(spec=multi_series_no_data)
        self.input_module.on_after_run(self._update_vw)
        self.dag_running()


stage_register["MultiSeries"] = MultiSeriesW
