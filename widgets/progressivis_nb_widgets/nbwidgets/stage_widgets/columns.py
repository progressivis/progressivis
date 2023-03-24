from .utils import (
    make_button,
    stage_register,
    dongle_widget,
    VBoxSchema,
    IpyHBoxSchema,
    SchemaBase,
)
import ipywidgets as ipw
import numpy as np
import operator as op
import weakref
from progressivis.table.repeater import Repeater, Computed
from progressivis.core import Sink, Module
from progressivis.table.compute import (
    week_day,
    UNCHANGED,
    make_if_else,
    ymd_string,
    is_weekend,
)

from typing import Any as AnyType, Optional, Tuple, List, Dict, Callable, Union

WidgetType = AnyType

DTYPES = [np.dtype(e).name for lst in np.sctypes.values() for e in lst] + ["datetime64"]  # type: ignore
UFUNCS: Dict[str, Callable[..., AnyType]] = {
    k: v for (k, v) in np.__dict__.items() if isinstance(v, np.ufunc) and v.nin == 1
}

ALL_FUNCS = UFUNCS.copy()
ALL_FUNCS.update(
    {"week_day": week_day, "is_weekend": is_weekend, "ymd_string": ymd_string}
)


class FuncW(ipw.VBox):
    def __init__(self, main: "PColumnsW", colname: str, fname: str) -> None:
        self._colname = colname
        self._fname = fname
        self._name = ipw.Text(
            value=f"{colname}_{fname}",
            placeholder="mandatory",
            description="Name:",
            disabled=False,
        )
        type_ = main.dtypes[colname]
        if type_ not in DTYPES:
            type_ = "object"
        self._dtype = ipw.Dropdown(
            value=type_,
            placeholder="dtype",
            options=DTYPES,
            description="Out dtype",
            ensure_option=True,
            disabled=False,
        )
        self._use = ipw.Checkbox(value=False, description="Use", disabled=False)
        self._use.observe(main.update_func_list, names="value")
        super().__init__([self._name, self._dtype, self._use])


class IfElseW(ipw.VBox):
    def __init__(self, main: "PColumnsW") -> None:
        self._main = weakref.ref(main)
        self._name = ipw.Text(
            value="", placeholder="mandatory", description="Name:", disabled=False
        )
        self._name.observe(self._name_cb, names="value")
        self._type = ipw.Dropdown(
            options=[("object", lambda x: x), ("integer", int), ("floating", float)],
            description="Type:",
            ensure_option=True,
            disabled=False,
        )
        self._name_box = ipw.HBox([self._name, self._type])
        self._is = ipw.Dropdown(
            options=[
                ("", None),
                (">", op.gt),
                ("<", op.lt),
                (">=", op.ge),
                ("<=", op.le),
                ("==", op.eq),
                ("NaN", np.isnan),
            ],
            description="Is",
            ensure_option=True,
            disabled=False,
        )
        self._is.observe(self._is_cb, names="value")
        self._than = ipw.Text(
            value="", placeholder="", description="Than:", disabled=False
        )
        self._than.observe(self._name_cb, names="value")
        self._cond_box = ipw.HBox([self._is, self._than])

        self._if_true_val = ipw.Text(
            value="", placeholder="mandatory", description="If True:", disabled=False
        )
        self._if_true_val.observe(self._name_cb, names="value")
        self._if_true_ck = ipw.Checkbox(
            value=False, description="Unchanged", disabled=False
        )
        self._if_true_ck.observe(self._if_true_ck_cb, names="value")
        self._if_true_box = ipw.HBox([self._if_true_val, self._if_true_ck])
        self._if_false_val = ipw.Text(
            value="", placeholder="mandatory", description="If False:", disabled=False
        )
        self._if_false_val.observe(self._name_cb, names="value")
        self._if_false_ck = ipw.Checkbox(
            value=False, description="Unchanged", disabled=False
        )
        self._if_false_ck.observe(self._if_false_ck_cb, names="value")
        self._if_false_box = ipw.HBox([self._if_false_val, self._if_false_ck])
        self._create_fnc = make_button("Create", disabled=True, cb=self._create_fnc_cb)
        super().__init__(
            [
                self._name_box,
                self._cond_box,
                self._if_true_box,
                self._if_false_box,
                self._create_fnc,
            ]
        )

    @property
    def main(self) -> Optional["PColumnsW"]:
        return self._main()

    def _name_cb(self, change: AnyType) -> None:
        self._check_all()

    def _is_cb(self, change: AnyType) -> None:
        if change["new"] is np.isnan:
            self._than.value = ""
            self._than.disabled = True
        else:
            self._than.disabled = False
        self._check_all()

    def _if_true_ck_cb(self, change: AnyType) -> None:
        if change["new"]:
            self._if_true_val.value = ""
            self._if_true_val.disabled = True
        else:
            self._if_true_val.disabled = False
        self._check_all()

    def _if_false_ck_cb(self, change: AnyType) -> None:
        if change["new"]:
            self._if_false_val.value = ""
            self._if_false_val.disabled = True
        else:
            self._if_false_val.disabled = False
        self._check_all()

    def _check_all(self) -> None:
        if not self._name.value:
            self._create_fnc.disabled = True
            return
        if not self._is.value:
            self._create_fnc.disabled = True
            return
        if not (self._if_true_val.value or self._if_true_ck.value):
            self._create_fnc.disabled = True
            return
        if not (self._if_false_val.value or self._if_false_ck.value):
            self._create_fnc.disabled = True
            return
        if self._if_true_ck.value and self._if_false_ck.value:
            self._create_fnc.disabled = True
            return
        if self._is.value is not np.isnan and not self._than.value:
            self._create_fnc.disabled = True
            return
        self._create_fnc.disabled = False

    def _create_fnc_cb(self, btn: AnyType) -> None:
        name = self._name.value
        assert name
        op_ = self._is.value
        assert op_ is not None
        conv_ = self._type.value
        than_ = None if op_ is np.isnan else conv_(self._than.value)
        if_true = (
            UNCHANGED if self._if_true_ck.value else conv_(self._if_true_val.value)
        )
        if_false = (
            UNCHANGED if self._if_false_ck.value else conv_(self._if_false_val.value)
        )
        func = make_if_else(op_=op_, test_val=than_, if_true=if_true, if_false=if_false)
        ALL_FUNCS.update({name: np.vectorize(func)})
        assert self.main is not None
        self.main.child.cols_funcs.child.funcs.options = [""] + list(ALL_FUNCS.keys())


class ColsFuncs(IpyHBoxSchema):
    class Schema(SchemaBase):
        cols: ipw.Select
        funcs: ipw.Select
        computed: Optional[FuncW]

    child: Schema


class KeepStored(IpyHBoxSchema):
    class Schema(SchemaBase):
        stored_cols: ipw.SelectMultiple
        keep_all: ipw.Checkbox

    child: Schema


class PColumnsW(VBoxSchema):
    class Schema(SchemaBase):
        custom_funcs: ipw.Accordion
        cols_funcs: ColsFuncs
        func_table: Optional[Union[ipw.Label, ipw.GridBox]]
        keep_stored: KeepStored
        btn_apply: ipw.Button

    child: Schema

    def init(self) -> None:
        self._col_widgets: Dict[Tuple[str, str], FuncW] = {}
        self._computed: List[Optional[FuncW]] = []
        self._numpy_ufuncs = ipw.Checkbox(
            value=True, description="Activate", disabled=False
        )
        self._numpy_ufuncs.observe(self._numpy_ufuncs_cb, names="value")
        self._if_else = IfElseW(self)
        self.child.custom_funcs = ipw.Accordion(
            children=[self._numpy_ufuncs, self._if_else], selected_index=None
        )
        self.child.custom_funcs.set_title(0, "Numpy universal functions")
        self.child.custom_funcs.set_title(1, "Add If-Else expressions")
        cols_t = [f"{c}:{t}" for (c, t) in self.dtypes.items()]
        col_list = list(zip(cols_t, self.dtypes.keys()))
        cols_funcs = ColsFuncs()
        cols_funcs.child.cols = ipw.Select(
            disabled=False, options=[("", "")] + col_list, rows=7
        )
        cols_funcs.child.cols.observe(self._columns_cb, names="value")
        cols_funcs.child.funcs = ipw.Select(
            disabled=True, options=[""] + list(ALL_FUNCS.keys()), rows=7
        )
        cols_funcs.child.funcs.observe(self._functions_cb, names="value")
        self.child.cols_funcs = cols_funcs
        keep_stored = KeepStored()
        keep_stored.child.stored_cols = ipw.SelectMultiple(
            options=col_list,
            value=[],
            rows=5,
            description="Keep also:",
            disabled=False,
        )
        keep_stored.child.keep_all = ipw.Checkbox(
            value=False, description="Select all", disabled=False
        )
        keep_stored.child.keep_all.observe(self._keep_all_cb, names="value")
        self.child.keep_stored = keep_stored
        self.child.btn_apply = make_button(
            "Apply", disabled=False, cb=self._btn_apply_cb
        )

    def _keep_all_cb(self, change: AnyType) -> None:
        val = change["new"]
        self.child.keep_stored.child.stored_cols.value = (
            list(self.dtypes.keys()) if val else []
        )

    def _numpy_ufuncs_cb(self, change: AnyType) -> None:
        if change["new"]:
            ALL_FUNCS.update(UFUNCS)
        else:
            for k in UFUNCS.keys():
                del ALL_FUNCS[k]
        self.child.cols_funcs.child.funcs.options = [""] + list(ALL_FUNCS.keys())

    def _columns_cb(self, change: AnyType) -> None:
        val = change["new"]
        self.child.cols_funcs.child.funcs.disabled = False
        if not val:
            self.child.cols_funcs.child.funcs.value = ""
            self.child.cols_funcs.child.funcs.disabled = True
            self.child.cols_funcs.child.computed = None
        elif self.child.cols_funcs.child.funcs.value:
            """key = (self._columns.value, self._functions.value)
            if key not in self._col_widgets:
                self._col_widgets[key] = FuncW(*key)
            self._computed_col = self._col_widgets[key]"""
            self.set_selection()
        else:
            self.child.cols_funcs.child.computed = None

    def _functions_cb(self, change: AnyType) -> None:
        val = change["new"]
        if not val:
            self.child.cols_funcs.child.computed = None
        else:
            self.set_selection()

    def set_selection(self) -> None:
        key = (
            self.child.cols_funcs.child.cols.value,
            self.child.cols_funcs.child.funcs.value,
        )
        if key not in self._col_widgets:
            self._col_widgets[key] = FuncW(self, *key)
        self.child.cols_funcs.child.computed = self._col_widgets[key]

    def _btn_apply_cb(self, btn: AnyType) -> None:
        """
        add_ufunc_column(self, name: str,
                         col: str,
                         ufunc: Callable,
                         dtype: Optional[np.dtype[Any]] = None,
                         xshape: Shape = ()) -> None:
        """
        comp = Computed()
        for (col, fname), wg in self._col_widgets.items():
            if not wg._use.value:
                continue
            func = ALL_FUNCS[fname]
            comp.add_ufunc_column(wg._name.value, col, func, np.dtype(wg._dtype.value))
        self.output_module = self.init_module(
            comp, columns=list(self.child.keep_stored.child.stored_cols.value)
        )
        self.make_chaining_box()
        self.dag_running()

    def init_module(
        self, computed: Computed, columns: Optional[List[str]] = None
    ) -> Repeater:
        s = self.input_module.scheduler()
        with s:
            rep = Repeater(computed=computed, columns=columns, scheduler=s)
            rep.input.table = self.input_module.output[self.input_slot]
            sink = Sink(scheduler=s)
            sink.input.inp = rep.output.result
            return rep

    def make_func_button(self, key: Tuple[str, str], wg: FuncW) -> ipw.Button:
        kcol, kfun = key

        def _cb(btn: AnyType) -> None:
            self.child.cols_funcs.child.cols.value = kcol
            self.child.cols_funcs.child.funcs.value = kfun

        btn = make_button(wg._name.value, cb=_cb)
        btn.layout = ipw.Layout(width="auto", height="40px")
        return btn

    def update_func_list(self, wg: FuncW) -> None:
        table_width = 4
        seld = {k: wg for (k, wg) in self._col_widgets.items() if wg._use.value}
        if not seld:
            self.child.func_table = None
            return
        lst = [self.make_func_button(key, wg) for (key, wg) in seld.items()]
        resume = table_width - len(lst) % table_width
        lst2 = [dongle_widget()] * resume
        self.child.func_table = ipw.GridBox(
            lst + lst2,
            layout=ipw.Layout(grid_template_columns=f"repeat({table_width}, 200px)"),
        )

    def get_underlying_modules(self) -> List[Module]:
        return [self.output_module]


stage_register["View"] = PColumnsW
