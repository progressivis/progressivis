from .utils import (
    make_button,
    stage_register,
    make_chaining_box,
    dongle_widget,
    set_child, ChainingWidget
)
import ipywidgets as ipw  # type: ignore
import pandas as pd
from progressivis.table.module import TableModule  # type: ignore
from progressivis.table.aggregate import Aggregate  # type: ignore
from progressivis.core import Sink  # type: ignore

from typing import (
    Any as AnyType,
    Optional,
    List,
    Tuple,
    Dict,
    Callable,
)

WidgetType = AnyType


def get_flag_status(dt: str, op: str) -> bool:
    return dt in ("string", "datetime64")  # op in type_op_mismatches.get(dt, set())


def _make_selm_obs(obj: "AggregateW") -> Callable:
    def _cbk(change: AnyType) -> None:
        obj.obs_flag = True
        cols = change["new"]
        for col in cols:
            obj.hidden_cols.remove(col)
            obj.visible_cols.append(col)
        assert obj._hidden_sel_wg
        obj._hidden_sel_wg.options = sorted(obj.hidden_cols)
        gb = obj.draw_matrix()
        set_child(obj, 1, gb)

    return _cbk


def _make_cbx_obs(obj: "AggregateW", col: str, func: str) -> Callable:
    def _cbk(change: AnyType) -> None:
        if func == "hide":
            obj.start_btn.disabled = True
            obj.hidden_cols.append(col)
            obj.visible_cols.remove(col)
            assert obj._hidden_sel_wg
            obj._hidden_sel_wg.options = sorted(obj.hidden_cols)
            gb = obj.draw_matrix()
            set_child(obj, 1, gb)
        else:
            obj.start_btn.disabled = False

    return _cbk


def _make_start_btn(obj):
    def _cbk(btn: ipw.Button) -> None:
        compute = [
            (col, fnc)
            for ((col, fnc), ck) in obj.info_cbx.items()
            if fnc != "hide" and ck.value
        ]
        obj._output_module = obj.init_aggregate(compute)
        obj._output_slot = "result"
        btn.disabled = True
        set_child(obj, 3, make_chaining_box(obj))
        obj.dag.requestAttention(obj.title, 'widget', "PROGRESS_NOTIFICATION", "0")
    return _cbk


class AggregateW(ipw.VBox, ChainingWidget):
    def __init__(
        self,
        parent: AnyType,
        dtypes: Dict[str, AnyType],
        input_module: TableModule,
        input_slot: str = "result", dag=None
    ) -> None:
        super().__init__(parent=parent,
                         dtypes=dtypes,
                         input_module=input_module,
                         input_slot=input_slot, dag=dag)
        self.dag_register()

        self.hidden_cols: List[str] = []
        fncs = ["hide"] + list(Aggregate.registry.keys())
        self.all_functions = dict(zip(fncs, fncs))
        self._hidden_sel_wg = ipw.SelectMultiple(
            options=self.hidden_cols, value=[], rows=5, description="❎", disabled=False,
        )
        self._hidden_sel_wg.observe(_make_selm_obs(self), "value")
        self.visible_cols: List[str] = list(self._dtypes.keys())
        self.obs_flag = False
        self.info_cbx: Dict[Tuple[str, str], ipw.Checkbox] = {}
        self._grid = self.draw_matrix()
        self.start_btn = make_button(
            "Activate", cb=_make_start_btn(self), disabled=True
        )
        self.children = (
            self._hidden_sel_wg,
            self._grid,
            self.start_btn,
            dongle_widget(),
        )

    def init_aggregate(self, compute: AnyType) -> Aggregate:
        s = self._input_module.scheduler()
        with s:
            aggr = Aggregate(compute=compute, scheduler=s)
            aggr.input.table = self._input_module.output[self._input_slot]
            sink = Sink(scheduler=s)
            sink.input.inp = aggr.output.result
            return aggr

    def draw_matrix(self, ext_df: Optional[pd.DataFrame] = None) -> ipw.GridBox:
        lst: List[WidgetType] = [ipw.Label("")] + [
            ipw.Label(s) for s in self.all_functions.values()
        ]
        width_ = len(lst)
        for col in sorted(self.visible_cols):
            col_type = self._dtypes[col]
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
        wgt.observe(_make_cbx_obs(self, col, func), "value")
        return wgt

    def get_underlying_modules(self):
        return [self._output_module]


stage_register["Aggregate"] = AggregateW
