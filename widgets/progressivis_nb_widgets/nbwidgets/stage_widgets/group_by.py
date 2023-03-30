from .utils import make_button, stage_register, dongle_widget, VBoxSchema, SchemaBase
import ipywidgets as ipw
from progressivis.table.group_by import (
    GroupBy,
    UTIME,
    DT_MAX,
    SubPColumn as SC,
    UTIME_SHORT_D,
)
from progressivis.core import Sink

from typing import Any as AnyType, Union, List, cast


def make_sel_multiple_dt(disabled: bool = True) -> ipw.SelectMultiple:
    return ipw.SelectMultiple(
        options=list(zip(UTIME, UTIME_SHORT_D.keys())),
        value=[],
        rows=DT_MAX,
        description="==>",
        disabled=disabled,
    )


class GroupByW(VBoxSchema):
    class Schema(SchemaBase):
        grouping_mode: Union[ipw.Label, ipw.RadioButtons]
        by_box: Union[ipw.SelectMultiple, ipw.HBox]
        start_btn: ipw.Button

    def init(self) -> None:
        self.child.grouping_mode = self.make_gr_mode()
        self.child.by_box = self.make_sel_multiple()
        self.child.start_btn = make_button(
            "Activate", cb=self._add_group_by_cb, disabled=True
        )

    def init_group_by(self, by: AnyType) -> GroupBy:
        s = self.input_module.scheduler()
        with s:
            grby = GroupBy(by=by, keepdims=True, scheduler=s)
            grby.input.table = self.input_module.output[self.input_slot]
            sink = Sink(scheduler=s)
            sink.input.inp = grby.output.result
            return grby

    def get_underlying_modules(self) -> List[str]:
        return [self.output_module.name]

    def _add_group_by_cb(self, btn: ipw.Button) -> None:
        self.child.grouping_mode.disabled = True
        self.child.by_box.disabled = True
        if self.child.grouping_mode.value == "columns":
            by = self.child.by_box.value
            assert by
            if len(by) == 1:
                by = by[0]
            self.child.by_box.disabled = True
        else:
            by_box = cast(ipw.HBox, self.child.by_box)
            dd, sel = by_box.children
            col = dd.value
            by = SC(col).dt["".join(sel.value)]
            by_box.children[0].disabled = True
            by_box.children[1].disabled = True
        self.output_module = self.init_group_by(by)
        self.output_slot = "result"
        btn.disabled = True
        self.make_chaining_box()
        self.dag_running()

    def _on_grouping_cb(self, val: AnyType) -> None:
        if val["new"] == "columns":
            self.child.by_box = self.make_sel_multiple()
        else:
            self.child.by_box = self.make_subcolumn_box()

    def make_gr_mode(self) -> Union[ipw.Label, ipw.RadioButtons]:
        wg: Union[ipw.Label, ipw.RadioButtons]
        if "datetime64" in self.input_dtypes.values():
            wg = ipw.RadioButtons(
                options=["columns", "datetime subcolumn", "multi index subcolumn"],
                description="Grouping mode:",
                disabled=False,
                style={"description_width": "initial"},
            )
            wg.observe(self._on_grouping_cb, names="value")
        else:
            wg = dongle_widget("columns")
        return wg

    def make_sel_multiple(self) -> ipw.SelectMultiple:
        selm = ipw.SelectMultiple(
            options=[(f"{col}:{t}", col) for (col, t) in self.dtypes.items()],
            value=[],
            rows=5,
            description="By",
            disabled=False,
        )

        def _f(val: AnyType) -> None:
            self.child.start_btn.disabled = not val["new"]

        selm.observe(_f, names="value")
        return selm

    def make_subcolumn_box(self) -> ipw.HBox:
        dd = ipw.Dropdown(
            options=[("", "")]
            + [
                (f"{col}:{t}", col)
                for (col, t) in self.dtypes.items()
                if t == "datetime64"
            ],
            value="",
            description="Datetime column:",
            disabled=False,
            style={"description_width": "initial"},
        )
        dt_sel = make_sel_multiple_dt()

        def _f(val: AnyType) -> None:
            if val["new"]:
                dt_sel.disabled = False
            else:
                dt_sel.disabled = True
                self.child.start_btn.disabled = True

        def _f_sel(val: AnyType) -> None:
            self.child.start_btn.disabled = not val["new"]

        dd.observe(_f, names="value")
        dt_sel.observe(_f_sel, names="value")
        return ipw.HBox([dd, dt_sel])


stage_register["Group by"] = GroupByW
