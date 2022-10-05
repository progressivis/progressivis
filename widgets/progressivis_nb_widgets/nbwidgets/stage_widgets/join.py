from .utils import (
    make_button,
    stage_register,
    make_chaining_box,
    dongle_widget,
    append_child, set_child, ChainingWidget,
    widget_by_key
)
import ipywidgets as ipw  # type: ignore
from progressivis.table.module import TableModule
from progressivis.table.group_by import UTIME_SHORT_D
from progressivis.table.join import Join
from progressivis.core import Sink
from typing import (
    Any as AnyType,
    Dict,
    Callable,
    List
)

WidgetType = AnyType
_l = ipw.Label


def _make_dd_cb(obj: "JoinW", col: str) -> Callable:
    def _cbk(change: AnyType) -> None:
        val = change["new"]
        ck, dd, mw = obj._primary_cols_dict[col]
        if val:
            ck.value = False
            ck.disabled = True
            obj._btn_start.disabled = False
            assert obj._primary_wg is not None
            if obj._primary_wg._output_dtypes[col] == "datetime64":
                mw.show()
            else:
                mw.hide()
        else:
            mw.hide()
            ck.disabled = False
            j = [dd for (_, dd, _) in obj._primary_cols_dict.values() if dd.value]
            if not j:
                obj._btn_start.disabled = True
    return _cbk


def _ck(name):
    return ipw.Checkbox(value=True,
                        description=name,
                        disabled=False,
                        indent=False)


class MaskWidget(ipw.HBox):
    def __init__(self):
        self.year = _ck("Y")
        self.month = _ck("M")
        self.day = _ck("D")
        self.hour = _ck("h")
        self.min_ = _ck("m")
        self.sec = _ck("s")
        self._ck_tpl = tuple([self.year, self.month, self.day, self.hour, self.min_, self.sec])
        super().__init__([])

    def show(self):
        self.children = self._ck_tpl

    def hide(self):
        for ck in self._ck_tpl:
            ck.value = True
        self.children = tuple([])

    def get_values(self):
        return [ck.value for ck in self._ck_tpl]

    def get_dt(self):
        return "".join([sym*ck.value for (sym, ck) in zip(UTIME_SHORT_D, self._ck_tpl)])


class JoinW(ipw.VBox, ChainingWidget):
    def __init__(
            self,
            parent: AnyType,
            dtypes: Dict[str, AnyType],
            input_module: TableModule,
            input_slot: str = "result",
            dag=None
    ) -> None:
        super().__init__(parent=parent,
                         dtypes=dtypes,
                         input_module=input_module,
                         input_slot=input_slot, dag=dag)
        self._output_dtypes = None
        dd_list = [(f"{k}[{n}]" if n else k, (k, n)) for (k, n) in widget_by_key.keys()]
        self._input_1 = _l(self.parent.title)
        self._role_1 = _l("primary")
        self._input_2 = ipw.Dropdown(  # type: ignore
            options=dd_list,
            description="",
            disabled=False,
            style={"description_width": "initial"},
        )
        self._role_2 = ipw.Dropdown(  # type: ignore
            options=["primary", "related"],
            value="related",
            description="",
            disabled=False,
            style={"description_width": "initial"},
        )
        self._cols_setup = ipw.Tab()
        self._primary_cols_dict: Dict[str, AnyType] = {}
        self._related_cols_dict: Dict[str, AnyType] = {}
        self._primary_wg = None
        self._related_wg = None
        self._role_2.observe(self._role_2_cb, "value")
        lst = [_l("Inputs"), _l("Roles"), self._input_1, self._role_1,
               self._input_2, self._role_2]
        gb = ipw.GridBox(
            lst,
            layout=ipw.Layout(grid_template_columns="50% 50%"),
        )
        gb.layout.border = "1px solid"  # type: ignore
        self._btn_ok = make_button("OK", cb=self._btn_ok_cb)
        self._how = ipw.Dropdown(  # type: ignore
            options=["inner", "outer"],
            value="inner",
            description="How",
            disabled=False,
            style={"description_width": "initial"},
        )
        self._btn_start = make_button("Start", disabled=True, cb=self._btn_start_cb)
        self.children = (gb, self._btn_ok, self._cols_setup, ipw.HBox([self._how, self._btn_start]), dongle_widget())

    def _btn_start_cb(self, btn):
        primary_cols = [k for (k, (ck, _, _)) in self._primary_cols_dict.items() if ck.value]
        related_cols = [k for (k, ck) in self._related_cols_dict.items() if ck.value]
        primary_on = [k for (k, (_, dd, _)) in self._primary_cols_dict.items() if dd.value]
        related_on = [dd.value for (_, dd, _) in self._primary_cols_dict.values() if dd.value]
        assert primary_on
        assert related_on
        assert len(primary_on) == len(related_on)
        primary_on = primary_on[0] if len(primary_on) == 1 else primary_on
        related_on = related_on[0] if len(related_on) == 1 else related_on
        s = self._input_module.scheduler()
        with s:
            inv_mask = None
            if (isinstance(primary_on,
                           str) and self._primary_wg._output_dtypes[
                               primary_on] == "datetime64"):
                _, _, mw = self._primary_cols_dict[primary_on]
                msk = mw.get_dt()
                if msk != "YMDhms":
                    inv_mask = msk
            join = Join(how=self._how.value, inv_mask=inv_mask, scheduler=s)
            join.create_dependent_modules(
                related_module=self._related_wg._output_module,
                primary_module=self._primary_wg._output_module,
                related_on=related_on,
                primary_on=primary_on,
                related_cols=related_cols,
                primary_cols=primary_cols
            )
            sink = Sink(scheduler=s)
            sink.input.inp = join.output.result
            self._output_module = join
        set_child(self, 4, make_chaining_box(self))
        self.dag.requestAttention(self.title, "widget", "PROGRESS_NOTIFICATION", 0)

    def _btn_ok_cb(self, k):
        self._input_2.disabled = True
        self._role_2.disabled = True
        self.dag.registerWidget(self, self.title, self.title, self.dom_id,
                                [self.parent.title,
                                 widget_by_key[self._input_2.value].title])
        if self._role_1.value == "primary":
            primary_wg = self.parent
            related_wg = widget_by_key[self._input_2.value]
        else:
            primary_wg = widget_by_key[self._input_2.value]
            related_wg = self.parent
        self._primary_wg = primary_wg
        self._related_wg = related_wg
        # primary cols
        ck_all = ipw.Checkbox(value=True,
                              description="",
                              disabled=False,
                              indent=False)
        lst: List[WidgetType] = [_l(""), _l("Keep"), _l("Join on"), _l("Subcolumns"),
                                 _l("*"), ck_all, _l(""), _l("")]
        for col, ty in primary_wg._output_dtypes.items():
            on_list = [""] + [c for (c, t) in related_wg._output_dtypes.items() if t == ty]
            ck = ipw.Checkbox(value=True,
                              description="",
                              disabled=False,
                              indent=False)
            dd = ipw.Dropdown(options=on_list,
                              description="",
                              disabled=False,
                              layout={"width": "initial"})
            dtw = MaskWidget()
            dd.observe(_make_dd_cb(self, col), "value")
            self._primary_cols_dict[col] = (ck, dd, dtw)
            lst.extend([_l(col), ck, dd, dtw])
        ck_all.observe(self._ck_all_cb, "value")
        p_gb = ipw.GridBox(
            lst,
            layout=ipw.Layout(grid_template_columns="20% 10% 40% 30%"),
        )
        append_child(self._cols_setup, p_gb, title="Primary")
        # related cols
        ck_all = ipw.Checkbox(value=True,
                              description="",
                              disabled=False,
                              indent=False)
        lst: List[WidgetType] = [_l(""), _l("Keep"),
                                 _l("*"), ck_all]
        for col in related_wg._output_dtypes.keys():
            ck = ipw.Checkbox(value=True,
                              description="",
                              disabled=False,
                              indent=False)
            self._related_cols_dict[col] = ck
            lst.extend([_l(col), ck])
        ck_all.observe(self._ck_all_cb2, "value")
        r_gb = ipw.GridBox(
            lst,
            layout=ipw.Layout(grid_template_columns="80% 20%"),
        )
        append_child(self._cols_setup, r_gb, title="Related")

    def _ck_all_cb(self, change: AnyType) -> None:
        val = change["new"]
        if val:
            for ck, dd, _ in self._primary_cols_dict.values():
                if not dd.value:
                    ck.value = True
        else:
            for ck, dd, _ in self._primary_cols_dict.values():
                ck.value = False

    def _ck_all_cb2(self, change: AnyType) -> None:
        val = change["new"]
        for ck in self._related_cols_dict.values():
            ck.value = val

    def _role_2_cb(self, change: Dict[str, AnyType]) -> None:
        role = change["new"]
        self._role_1.value = "primary" if role == "related" else "related"


stage_register["Join"] = JoinW
