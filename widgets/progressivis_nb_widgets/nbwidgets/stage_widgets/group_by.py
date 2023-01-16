from .utils import (
    make_button,
    stage_register,
    dongle_widget,
    VBoxSchema
)
import ipywidgets as ipw  # type: ignore
from progressivis.table.group_by import (GroupBy, UTIME, DT_MAX,  # type: ignore
                                         SubPColumn as SC, UTIME_SHORT_D)
from progressivis.core import Sink  # type: ignore

from typing import (
    Any as AnyType,
)

WidgetType = AnyType


def make_sel_multiple_dt(disabled: bool = True) -> WidgetType:
    return ipw.SelectMultiple(
        options=list(zip(UTIME, UTIME_SHORT_D.keys())),
        value=[],
        rows=DT_MAX,
        description="==>",
        disabled=disabled,
    )


class GroupByW(VBoxSchema):
    def init(self) -> None:
        self.schema = dict(
            grouping_mode=self.make_gr_mode(),
            by_box=self.make_sel_multiple(),
            start_btn=make_button(
                "Activate", cb=self._add_group_by_cb, disabled=True
            )
        )

    def init_group_by(self, by: AnyType) -> GroupBy:
        s = self.input_module.scheduler()
        with s:
            grby = GroupBy(by=by, keepdims=True, scheduler=s)
            grby.input.table = self.input_module.output[self.input_slot]
            sink = Sink(scheduler=s)
            sink.input.inp = grby.output.result
            return grby

    def get_underlying_modules(self):
        return [self._output_module]

    def _add_group_by_cb(self, btn: ipw.Button) -> None:
        self["grouping_mode"].disabled = True
        self["by_box"].disabled = True
        if self["grouping_mode"].value == "columns":
            by = self["by_box"].value
            assert by
            if len(by) == 1:
                by = by[0]
            self["by_box"].disabled = True
        else:
            dd, sel = self["by_box"].children
            col = dd.value
            by = SC(col).dt["".join(sel.value)]
            self["by_box"].children[0].disabled = True
            self["by_box"].children[1].disabled = True
        self.output_module = self.init_group_by(by)
        self.output_slot = "result"
        btn.disabled = True
        self.make_chaining_box()
        self.dag_running()

    def _on_grouping_cb(self, val: AnyType) -> None:
        if val["new"] == "columns":
            self["by_box"] = self.make_sel_multiple()
        else:
            self["by_box"] = self.make_subcolumn_box()

    def make_gr_mode(self) -> WidgetType:
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

    def make_sel_multiple(self) -> WidgetType:
        selm = ipw.SelectMultiple(
            options=[(f"{col}:{t}", col) for (col, t) in self.dtypes.items()],
            value=[], rows=5, description="By", disabled=False,
        )

        def _f(val):
            self["start_btn"].disabled = not val["new"]

        selm.observe(_f, names="value")
        return selm

    def make_subcolumn_box(self) -> WidgetType:
        dd = ipw.Dropdown(
            options=[("", "")]
            + [(f"{col}:{t}", col)
               for (col, t) in self.dtypes.items()
               if t == "datetime64"],
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
                self["start_btn"].disabled = True

        def _f_sel(val):
            self["start_btn"].disabled = not val["new"]

        dd.observe(_f, names="value")
        dt_sel.observe(_f_sel, names="value")
        return ipw.HBox([dd, dt_sel])


stage_register["Group by"] = GroupByW
