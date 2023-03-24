from .utils import make_button, stage_register, VBoxSchema, SchemaBase
import ipywidgets as ipw
import pandas as pd
from progressivis.table.aggregate import Aggregate
from progressivis.core import Sink, Module

from typing import Any as AnyType, Optional, List, Tuple, Dict, Callable

WidgetType = AnyType


def get_flag_status(dt: str, op: str) -> bool:
    return dt in ("string", "datetime64")  # op in type_op_mismatches.get(dt, set())


class AggregateW(VBoxSchema):
    class Schema(SchemaBase):
        hidden_sel: ipw.SelectMultiple
        grid: ipw.GridBox
        start_btn: ipw.Button

    child: Schema

    def init(self) -> None:
        self.hidden_cols: List[str] = []
        fncs = ["hide"] + list(Aggregate.registry.keys())
        self.all_functions = dict(zip(fncs, fncs))
        self.child.hidden_sel = ipw.SelectMultiple(
            options=self.hidden_cols,
            value=[],
            rows=5,
            description="❎",
            disabled=False,
        )
        self.child.hidden_sel.observe(self._selm_obs_cb, "value")
        self.visible_cols: List[str] = list(self.dtypes.keys())
        self.obs_flag = False
        self.info_cbx: Dict[Tuple[str, str], ipw.Checkbox] = {}
        self.child.grid = self.draw_matrix()
        self.child.start_btn = make_button(
            "Activate", cb=self._start_btn_cb, disabled=True
        )

    def init_aggregate(self, compute: AnyType) -> Aggregate:
        s = self.input_module.scheduler()
        with s:
            aggr = Aggregate(compute=compute, scheduler=s)
            aggr.input.table = self.input_module.output[self.input_slot]
            sink = Sink(scheduler=s)
            sink.input.inp = aggr.output.result
            return aggr

    def draw_matrix(self, ext_df: Optional[pd.DataFrame] = None) -> ipw.GridBox:
        lst: List[WidgetType] = [ipw.Label("")] + [
            ipw.Label(s) for s in self.all_functions.values()
        ]
        width_ = len(lst)
        for col in sorted(self.visible_cols):
            col_type = self.dtypes[col]
            lst.append(ipw.Label(f"{col}:{col_type}"))
            for k in self.all_functions.keys():
                lst.append(self._info_checkbox(col, k, get_flag_status(col_type, k)))
        gb = ipw.GridBox(
            lst,
            layout=ipw.Layout(grid_template_columns=f"200px repeat({width_-1}, 70px)"),
        )
        return gb

    def _info_checkbox(self, col: str, func: str, dis: bool) -> ipw.Checkbox:
        wgt = ipw.Checkbox(value=False, description="", disabled=dis, indent=False)
        self.info_cbx[(col, func)] = wgt
        wgt.observe(self._make_cbx_obs(col, func), "value")
        return wgt

    def _start_btn_cb(self, btn: ipw.Button) -> None:
        compute = [
            (col, fnc)
            for ((col, fnc), ck) in self.info_cbx.items()
            if fnc != "hide" and ck.value
        ]
        self.output_module = self.init_aggregate(compute)
        self.output_slot = "result"
        btn.disabled = True
        self.make_chaining_box()
        self.dag_running()

    def _selm_obs_cb(self, change: AnyType) -> None:
        self.obs_flag = True
        cols = change["new"]
        for col in cols:
            self.hidden_cols.remove(col)
            self.visible_cols.append(col)
        self.child.hidden_sel.options = sorted(self.hidden_cols)
        self.child.grid = self.draw_matrix()

    def _make_cbx_obs(self, col: str, func: str) -> Callable[[AnyType], None]:
        def _cbk(change: AnyType) -> None:
            if func == "hide":
                self.child.start_btn.disabled = True
                self.hidden_cols.append(col)
                self.visible_cols.remove(col)
                self.child.hidden_sel.options = sorted(self.hidden_cols)
                self.child.grid = self.draw_matrix()
            else:
                self.child.start_btn.disabled = False

        return _cbk

    def get_underlying_modules(self) -> List[Module]:
        return [self.output_module]


stage_register["Aggregate"] = AggregateW
