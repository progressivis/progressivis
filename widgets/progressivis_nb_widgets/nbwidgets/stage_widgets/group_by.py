from .utils import (
    make_button,
    stage_register,
    make_chaining_box,
    dongle_widget,
    set_child,
)
import ipywidgets as ipw
from progressivis.table.module import TableModule
from progressivis.table.group_by import GroupBy, UTIME, DT_MAX, SubColumn as SC
from progressivis.core import Sink

from typing import (
    Any as AnyType,
    Dict,
    Callable,
)

WidgetType = AnyType


def undecorate(typed_col: str) -> str:
    return typed_col.split(":")[0]


def make_add_group_by(obj: "GroupByW") -> Callable:
    def _cbk(btn: ipw.Button) -> None:
        obj.grouping_mode.disabled = True
        obj.by_box.disabled = True
        if obj.grouping_mode.value == "columns":
            by = [undecorate(c) for c in obj.by_box.value]
            assert by
            if len(by) == 1:
                by = by[0]
            obj.by_box.disabled = True
        else:
            dd, sel = obj.by_box.children
            col = dd.value
            idx = sel.value
            by = SC(undecorate(col)).ix[list(idx)]
            obj.by_box.children[0].disabled = True
            obj.by_box.children[1].disabled = True
        obj._output_module = obj.init_group_by(by)
        obj._output_slot = "result"
        btn.disabled = True
        set_child(obj, 3, make_chaining_box(obj))

    return _cbk


def make_subcolumn_box(obj: "GroupByW") -> WidgetType:
    dd = ipw.Dropdown(
        options=[""]
        + [f"{col}:{t}" for (col, t) in obj._dtypes.items() if t == "datetime64"],
        value="",
        description="Datetime column:",
        disabled=False,
        style={"description_width": "initial"},
    )
    dt_sel = make_sel_multiple_dt()

    def _f(val):
        if val["new"]:
            dt_sel.disabled = False
        else:
            dt_sel.disabled = True
            obj.start_btn.disabled = True

    def _f_sel(val):
        if val["new"]:
            obj.start_btn.disabled = False
        else:
            obj.start_btn.disabled = True

    dd.observe(_f, names="value")
    dt_sel.observe(_f_sel, names="value")
    return ipw.HBox([dd, dt_sel])


def make_sel_multiple(obj: "GroupByW"):
    selm = ipw.SelectMultiple(
        options=obj._dtypes.keys(), value=[], rows=5, description="By", disabled=False,
    )

    def _f(val):
        if val["new"]:
            obj.start_btn.disabled = False
        else:
            obj.start_btn.disabled = True

    selm.observe(_f, names="value")
    return selm


def make_sel_multiple_dt(disabled: bool = True) -> WidgetType:
    return ipw.SelectMultiple(
        options=list(zip(UTIME, range(len(UTIME)))),
        value=[],
        rows=DT_MAX,
        description="==>",
        disabled=disabled,
    )


def make_on_grouping(obj: "GroupByW") -> Callable:
    def _fnc(val: AnyType) -> None:
        if val["new"] == "columns":
            obj.by_box = make_sel_multiple(obj)
        else:
            obj.by_box = make_subcolumn_box(obj)
        # obj.children = (obj.grouping_mode, obj.by_box, obj.start_btn)
        set_child(obj, 1, obj.by_box)

    return _fnc


def make_gr_mode(obj: "GroupByW"):
    if "datetime64" in obj._dtypes.values():
        wg = ipw.RadioButtons(
            options=["columns", "datetime subcolumn", "multi index subcolumn"],
            description="Grouping mode:",
            disabled=False,
            style={"description_width": "initial"},
        )
        wg.observe(make_on_grouping(obj), names="value")
    else:
        wg = dongle_widget("columns")
    obj.start_btn.disabled = True
    return wg


class GroupByW(ipw.VBox):
    def __init__(
        self,
        frame: AnyType,
        dtypes: Dict[str, AnyType],
        input_module: TableModule,
        input_slot: str = "result",
    ) -> None:
        super().__init__()
        self._frame = frame
        self._dtypes = dtypes
        self._input_module = input_module
        self._input_slot = input_slot
        self.start_btn = make_button(
            "Activate", cb=make_add_group_by(self), disabled=True
        )
        self.grouping_mode = make_gr_mode(self)
        self.by_box = make_sel_multiple(self)
        self.children = (
            self.grouping_mode,
            self.by_box,
            self.start_btn,
            dongle_widget(),
        )

    def init_group_by(self, by: AnyType) -> GroupBy:
        s = self._input_module.scheduler()
        with s:
            grby = GroupBy(by=by, scheduler=s)
            grby.input.table = self._input_module.output[self._input_slot]
            sink = Sink(scheduler=s)
            sink.input.inp = grby.output.result
            return grby


stage_register["Group by"] = GroupByW
