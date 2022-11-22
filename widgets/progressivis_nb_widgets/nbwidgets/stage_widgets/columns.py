from .utils import (
    make_button,
    stage_register,
    make_chaining_box,
    dongle_widget,
    set_child, ChainingVBox
)
import ipywidgets as ipw  # type: ignore
import numpy as np
import operator as op
from progressivis.table.repeater import Repeater, Computed  # type: ignore
from progressivis.core import Sink  # type: ignore
from progressivis.table.compute import (week_day, UNCHANGED,
                                        make_if_else, ymd_string, is_weekend)

from typing import (
    Any as AnyType,
    Optional,
    List,
    Dict,
)

WidgetType = AnyType
_l = ipw.Label
_dw = dongle_widget

DTYPES = [np.dtype(e).name for lst in np.sctypes.values() for e in lst] + ["datetime64"]
UFUNCS = {
    k: v for (k, v) in np.__dict__.items() if isinstance(v, np.ufunc) and v.nin == 1
}

ALL_FUNCS = UFUNCS.copy()
ALL_FUNCS.update({"week_day": week_day, "is_weekend": is_weekend, "ymd_string": ymd_string})


class FuncW(ipw.VBox):
    def __init__(self, main, colname, fname):
        self._main = main
        self._colname = colname
        self._fname = fname
        self._name = ipw.Text(
            value=f"{colname}_{fname}", placeholder="mandatory", description="Name:",
            disabled=False
            )
        type_ = self._main._dtypes[colname]
        if type_ not in DTYPES:
            type_ = "object"
        self._dtype = ipw.Dropdown(
            value=type_,
            placeholder="dtype",
            options=DTYPES,
            description="Out dtype",
            ensure_option=True,
            disabled=False
        )
        self._use = ipw.Checkbox(value=False, description="Use",
                                 disabled=False)
        self._use.observe(self._main.update_func_list, names="value")
        super().__init__([self._name, self._dtype, self._use])


class IfElseW(ipw.VBox):
    def __init__(self, main):
        self._main = main
        self._name = ipw.Text(
            value="", placeholder="mandatory", description="Name:",
            disabled=False
            )
        self._name.observe(self._name_cb, names="value")
        self._type = ipw.Dropdown(
            options=[("object", lambda x: x), ("integer", int), ("floating", float)],
            description="Type:",
            ensure_option=True,
            disabled=False
        )
        self._name_box = ipw.HBox([self._name, self._type])
        self._is = ipw.Dropdown(
            options=[("", None), (">", op.gt),
                     ("<", op.lt), (">=", op.ge),
                     ("<=", op.le), ("==", op.eq), ("NaN", np.isnan)],
            description="Is",
            ensure_option=True,
            disabled=False
        )
        self._is.observe(self._is_cb, names="value")
        self._than = ipw.Text(
            value="", placeholder="", description="Than:",
            disabled=False
            )
        self._than.observe(self._name_cb, names="value")
        self._cond_box = ipw.HBox([self._is, self._than])

        self._if_true_val = ipw.Text(
            value="", placeholder="mandatory", description="If True:",
            disabled=False
            )
        self._if_true_val.observe(self._name_cb, names="value")
        self._if_true_ck = ipw.Checkbox(value=False, description="Unchanged",
                                        disabled=False)
        self._if_true_ck.observe(self._if_true_ck_cb, names="value")
        self._if_true_box = ipw.HBox([self._if_true_val, self._if_true_ck])
        self._if_false_val = ipw.Text(
            value="", placeholder="mandatory", description="If False:",
            disabled=False
            )
        self._if_false_val.observe(self._name_cb, names="value")
        self._if_false_ck = ipw.Checkbox(value=False, description="Unchanged",
                                         disabled=False)
        self._if_false_ck.observe(self._if_false_ck_cb, names="value")
        self._if_false_box = ipw.HBox([self._if_false_val, self._if_false_ck])
        self._create_fnc = make_button(
            "Create", disabled=True, cb=self._create_fnc_cb
        )
        super().__init__([self._name_box, self._cond_box,
                          self._if_true_box, self._if_false_box,
                          self._create_fnc])

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

    def _check_all(self):
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

    def _create_fnc_cb(self, btn):
        name = self._name.value
        assert name
        op_ = self._is.value
        assert op_ is not None
        conv_ = self._type.value
        than_ = None if op_ is np.isnan else conv_(self._than.value)
        if_true = UNCHANGED if self._if_true_ck.value else conv_(self._if_true_val.value)
        if_false = UNCHANGED if self._if_false_ck.value else conv_(self._if_false_val.value)
        func = make_if_else(op_=op_, test_val=than_, if_true=if_true, if_false=if_false)
        ALL_FUNCS.update({name: np.vectorize(func)})
        self._main._functions.options = [""] + list(ALL_FUNCS.keys())


class ColumnsW(ChainingVBox):
    def __init__(self, ctx: Dict[str, AnyType]) -> None:
        super().__init__(ctx)
        self._col_widgets = {}
        self._computed = []
        self._numpy_ufuncs = ipw.Checkbox(value=True,
                                          description="Activate",
                                          disabled=False)
        self._numpy_ufuncs.observe(self._numpy_ufuncs_cb, names="value")
        self._if_else = IfElseW(self)
        self._custom_funcs = ipw.Accordion(children=[self._numpy_ufuncs, self._if_else], selected_index=None)
        self._custom_funcs.set_title(0, "Numpy universal functions")
        self._custom_funcs.set_title(1, "Add If-Else expressions")
        cols_t = [f"{c}:{t}" for (c, t) in self._dtypes.items()]
        col_list = list(zip(cols_t, self._dtypes.keys()))
        self._columns = ipw.Select(disabled=False,
                                   options=[("", "")]+col_list, rows=7)
        self._columns.observe(self._columns_cb, names="value")
        self._functions = ipw.Select(disabled=True,
                                     options=[""] +
                                     list(ALL_FUNCS.keys()),
                                     rows=7)
        self._functions.observe(self._functions_cb, names="value")
        self._computed_col = _dw()
        self._hbox = ipw.HBox([self._columns, self._functions, self._computed_col])
        self._func_table = _dw()
        self._stored_cols = ipw.SelectMultiple(
            options=col_list,
            value=[], rows=5, description="Keep also:", disabled=False,
        )
        self._keep_all = ipw.Checkbox(value=False,
                                      description="Select all",
                                      disabled=False)
        self._keep_all.observe(self._keep_all_cb, names="value")
        self._btn_apply = self._btn_ok = make_button(
            "Apply", disabled=False, cb=self._btn_apply_cb
        )
        self._chaining_box = _dw()
        self.children = (self._custom_funcs, self._hbox, self._func_table,
                         ipw.HBox([self._stored_cols, self._keep_all]),
                         self._btn_apply,
                         self._chaining_box)

    def _keep_all_cb(self, change: AnyType) -> None:
        val = change["new"]
        if val:
            self._stored_cols.value = list(self._dtypes.keys())
        else:
            self._stored_cols.value = []

    def _numpy_ufuncs_cb(self, change: AnyType) -> None:
        if change["new"]:
            ALL_FUNCS.update(UFUNCS)
        else:
            for k in UFUNCS.keys():
                del ALL_FUNCS[k]
        self._functions.options = [""] + list(ALL_FUNCS.keys())

    def _columns_cb(self, change: AnyType) -> None:
        val = change["new"]
        self._functions.disabled = False
        if not val:
            self._functions.value = ""
            self._functions.disabled = True
            self._computed_col = _dw()
        elif self._functions.value:
            """key = (self._columns.value, self._functions.value)
            if key not in self._col_widgets:
                self._col_widgets[key] = FuncW(*key)
            self._computed_col = self._col_widgets[key]"""
            self.set_selection()
        else:
            self._computed_col = _dw()

    def _functions_cb(self, change: AnyType) -> None:
        val = change["new"]
        if not val:
            self._computed_col = _dw()
        else:
            self.set_selection()

    def set_selection(self):
        key = (self._columns.value, self._functions.value)
        if key not in self._col_widgets:
            self._col_widgets[key] = FuncW(self, *key)
        self._computed_col = self._col_widgets[key]
        set_child(self._hbox, 2, self._computed_col)

    def _btn_apply_cb(self, btn):
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
        self._output_module = self.init_module(comp, columns=list(self._stored_cols.value))
        set_child(self, 5, make_chaining_box(self))
        self.dag_running()

    def init_module(self, computed: Computed, columns: Optional[List[str]] = None) -> Repeater:
        s = self._input_module.scheduler()
        with s:
            rep = Repeater(computed=computed, columns=columns, scheduler=s)
            rep.input.table = self._input_module.output[self._input_slot]
            sink = Sink(scheduler=s)
            sink.input.inp = rep.output.result
            return rep

    def make_func_button(self, key, wg):
        kcol, kfun = key

        def _cb(btn):
            self._columns.value = kcol
            self._functions.value = kfun
        btn = make_button(wg._name.value, cb=_cb)
        btn.layout = ipw.Layout(width='auto', height='40px')
        return btn

    def update_func_list(self, wg):
        table_width = 4
        seld = {k: wg for (k, wg) in self._col_widgets.items() if wg._use.value}
        if not seld:
            self._func_table = _dw()
            return
        lst = [self.make_func_button(key, wg) for (key, wg) in seld.items()]
        resume = table_width - len(lst) % table_width
        lst2 = [_dw()] * resume
        self._func_table = ipw.GridBox(
            lst+lst2,
            layout=ipw.Layout(grid_template_columns=f"repeat({table_width}, 200px)"),
        )
        set_child(self, 2, self._func_table)

    def get_underlying_modules(self):
        return [self._output_module]


stage_register["View"] = ColumnsW
