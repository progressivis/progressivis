from functools import singledispatch
from collections import Iterable
from itertools import product
import ipywidgets as ipw
import numpy as np
import pandas as pd
from progressivis.core import asynchronize, aio
from vega.widget import VegaWidget
from ._hist1d_schema import hist1d_spec_no_data, kll_spec_no_data
from ._corr_schema import corr_spec_no_data
from ._bar_schema import bar_spec_no_data
from collections import defaultdict

# https://stackoverflow.com/questions/59741643/how-to-specify-rule-line-with-a-single-value-in-vegalite

N_BINS = 128
LAST_BIN = N_BINS - 1


wg_lock = aio.Lock()
IS_DEBUG = True
debug_console = None
range_widgets = {}


def bins_range_slider(desc):
    return ipw.IntRangeSlider(
        value=[0, LAST_BIN],
        min=0,
        max=LAST_BIN,
        step=1,
        description=desc,
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=False,
        readout_format="d",
    )


def _make_cbx_obs(dyn_viewer, col, func):
    def _cbk(change):
        dyn_viewer._btn_apply.disabled = False
        if func == "hide":
            dyn_viewer.hidden_cols.append(col)
            dyn_viewer.visible_cols.remove(col)
            dyn_viewer._hidden_sel_wg.options = sorted(dyn_viewer.hidden_cols)
            gb = dyn_viewer.draw_matrix()
            dyn_viewer.unlock_conf()
            dyn_viewer.conf_box.children = (
                dyn_viewer._hidden_sel_wg,
                gb,
                dyn_viewer._btn_bar,
            )

    return _cbk


def _make_btn_edit_cb(dyn_viewer):
    def _cbk(btn):
        btn.disabled = True
        dyn_viewer.save_for_cancel = (
            dyn_viewer.hidden_cols[:],
            dyn_viewer.visible_cols[:],
            dyn_viewer.matrix_to_df(),
        )
        dyn_viewer._btn_cancel.disabled = False
        dyn_viewer._btn_apply.disabled = True
        dyn_viewer.unlock_conf()

    return _cbk


def _make_btn_cancel_cb(dyn_viewer):
    def _cbk(btn):
        btn.disabled = True
        dyn_viewer._btn_edit.disabled = False
        hcols, vcols, df = dyn_viewer.save_for_cancel
        dyn_viewer.hidden_cols = hcols[:]
        dyn_viewer.visible_cols = vcols[:]
        dyn_viewer._hidden_sel_wg.options = dyn_viewer.hidden_cols
        gb = dyn_viewer.draw_matrix(df)
        dyn_viewer.lock_conf()
        dyn_viewer.conf_box.children = (
            dyn_viewer._hidden_sel_wg,
            gb,
            dyn_viewer._btn_bar,
        )
        dyn_viewer.lock_conf()
        dyn_viewer._btn_apply.disabled = True

    return _cbk


def _make_btn_apply_cb(dyn_viewer):
    def _cbk(btn):
        btn.disabled = True
        dyn_viewer._btn_edit.disabled = False
        dyn_viewer._btn_cancel.disabled = True
        dyn_viewer.lock_conf()

        async def _coro():
            dyn_viewer._last_df = dyn_viewer.matrix_to_df()
            await dyn_viewer._registry_mod.variable.from_input(
                {
                    "matrix": dyn_viewer._last_df,
                    "hidden_cols": dyn_viewer.hidden_cols[:],
                }
            )

        aio.create_task(_coro())

    return _cbk


def _make_selm_obs(dyn_viewer):
    def _cbk(change):
        if dyn_viewer.obs_flag:
            return
        try:
            dyn_viewer.obs_flag = True
            cols = change["new"]
            for col in cols:
                dyn_viewer.hidden_cols.remove(col)
                dyn_viewer.visible_cols.append(col)
            dyn_viewer._hidden_sel_wg.options = sorted(dyn_viewer.hidden_cols)
            gb = dyn_viewer.draw_matrix()
            dyn_viewer.unlock_conf()
            dyn_viewer.conf_box.children = (
                dyn_viewer._hidden_sel_wg,
                gb,
                dyn_viewer._btn_bar,
            )
        finally:
            dyn_viewer.obs_flag = False

    return _cbk


def make_button(label, disabled=False, cb=None):
    btn = ipw.Button(
        description=label,
        disabled=disabled,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip=label,
        icon="check",  # (FontAwesome names without the `fa-` prefix)
    )
    if cb is not None:
        btn.on_click(cb)
    return btn


def make_observer(hname, sk_mod, lower_mod, upper_mod):
    def _observe_range(val):
        async def _coro(v):
            async with wg_lock:
                lo = v["new"][0]
                up = v["new"][1]
                res = sk_mod.result
                hist = res["pmf"]
                min_ = res["min"]
                max_ = res["max"]
                len_ = len(hist)
                bins_ = np.linspace(min_, max_, len_)
                lower = bins_[lo]
                upper = bins_[up]
                await lower_mod.from_input({hname: lower})
                await upper_mod.from_input({hname: upper})

        aio.create_task(_coro(val))

    return _observe_range


def my_print(*args, **kw):
    if debug_console:
        with debug_console:
            print(*args, **kw)


once_dict = {}


def my_print_once(*args):
    if args in once_dict:
        return
    once_dict[args] = True
    my_print(*args)


nn_dict = defaultdict(int)


def my_print_n(*args, repeat=1, **kw):
    if nn_dict[args] >= repeat:
        return
    nn_dict[args] += 1
    my_print(*args, kw)


def corr_as_vega_dataset(mod, columns=None):
    """ """
    if columns is None:
        columns = mod._columns

    def _c(kx, ky):
        return mod.result[frozenset([kx, ky])]

    return [
        dict(corr=_c(kx, ky), corr_label=f"{_c(kx,ky):.2f}", var=kx, var2=ky)
        for (kx, ky) in product(columns, columns)
    ]


def categ_as_vega_dataset(categs):
    return [{"category": k, "count": v} for (k, v) in categs.items()]


def _refresh_info(wg):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(wg.refresh_info)

    return _coro


@singledispatch
def format_label(arg):
    return str(arg)


@format_label.register(float)
@format_label.register(np.floating)
def _(arg):
    return f"{arg:.4f}"


@format_label.register(str)
def _(arg):
    return arg


@format_label.register(Iterable)
def _(arg):
    return str(len(arg))


def refresh_info_sketch(hout, hmod):
    if not hmod.result:
        return
    res = hmod.result
    hist = res["pmf"]
    min_ = res["min"]
    max_ = res["max"]
    len_ = len(hist)
    bins_ = np.linspace(min_, max_, len_)
    rule_lower = np.zeros(len_, dtype="int32")
    rule_upper = np.zeros(len_, dtype="int32")
    range_widget = range_widgets.get(hmod.column)
    if range_widget is not None:
        rule_lower[0] = range_widget.value[0]
        rule_upper[0] = range_widget.value[1]
    source = pd.DataFrame(
        {
            "xvals": bins_,
            "nbins": range(len(hist)),
            "rule_lower": rule_lower,
            "rule_upper": rule_upper,
            "level": hist,
        }
    )
    hout.children[0].children[0].update("data", remove="true", insert=source)
    # range slider, labels etc.
    if range_widget is None:
        return
    # range_widget = hout.children[0].children[1].children[0]
    label_min = hout.children[0].children[1].children[1]
    label_max = hout.children[0].children[1].children[2]
    label_min.value = f"{bins_[range_widget.value[0]]:.2f}"
    label_max.value = f" -- {bins_[range_widget.value[1]]:.2f}"


def refresh_info_barplot(hout, hmod):
    categs = hmod.result
    if not categs:
        return
    dataset = categ_as_vega_dataset(categs)
    hout.update("data", remove="true", insert=dataset)
    return


def refresh_info_hist_1d(hout, h1d_mod):
    if not h1d_mod.result:
        return
    res = h1d_mod.result.last().to_dict()
    hist = res["array"]
    min_ = res["min"]
    max_ = res["max"]
    bins_ = np.linspace(min_, max_, len(hist))
    source = pd.DataFrame({"xvals": bins_, "nbins": range(len(hist)), "level": hist})
    hout.children[1].update("data", remove="true", insert=source)


def _refresh_info_sketch(hout, hmod):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(refresh_info_sketch, hout, hmod)

    return _coro


def _refresh_info_barplot(hout, hmod):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(refresh_info_barplot, hout, hmod)

    return _coro


def _refresh_info_hist_1d(hout, hmod):
    async def _coro(_1, _2):
        _ = _1, _2
        # async with wg_lock:
        await asynchronize(refresh_info_hist_1d, hout, hmod)

    return _coro


def refresh_info_corr(cout, cmod, main_wg):
    if not cmod.result:
        return
    cols = cmod._columns  # main_wg.get_corr_sel()
    dataset = corr_as_vega_dataset(cmod, cols)
    cout.update("data", remove="true", insert=dataset)


def _refresh_info_corr(cout, cmod, wg):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(refresh_info_corr, cout, cmod, wg)

    return _coro


type_op_mismatches = dict(string=set(["min", "max", "var", "corr"]))


def get_flag_status(dt, op):
    return op in type_op_mismatches.get(dt, set())


class DynTab(ipw.Tab):
    def set_next_title(self, name):
        pos = len(self.children) - 1
        self.set_title(pos, name)

    def get_titles(self):
        return [self.get_title(pos) for pos in range(len(self.children))]

    def set_tab(self, title, wg, overwrite=True):
        all_titles = self.get_titles()
        if title in all_titles:
            if not overwrite:
                return
            pos = all_titles.index(title)
            children_ = list(self.children)
            children_[pos] = wg
            self.children = tuple(children_)
        else:
            self.children += (wg,)
            self.set_next_title(title)

    def remove_tab(self, title):
        all_titles = self.get_titles()
        if title not in all_titles:
            return
        pos = all_titles.index(title)
        children_ = list(self.children)
        children_ = children_[:pos] + children_[pos + 1 :]
        titles_ = all_titles[:pos] + all_titles[pos + 1 :]
        self.children = tuple(children_)
        for i, t in enumerate(titles_):
            self.set_title(i, t)


class DynViewer(DynTab):
    # scalar_functions = {'min': 'Min', 'max': 'Max'}
    def __init__(self, dshape_mod, registry_mod, scheduler=None):
        self._dshape_mod = dshape_mod
        self._registry_mod = registry_mod
        self.hidden_cols = []
        self._hidden_sel_wg = None
        self.visible_cols = None
        self._last_df = None
        self.previous_visible_cols = []
        self.info_labels = {}
        self.info_cbx = {}
        self._hdict = {}
        self._hist_tab = None
        self._hist_sel = set()
        self._corr_sel = set()
        # self._dshape_mod.on_after_run(_refresh_info(self))
        self._dshape_mod.scheduler().on_loop(_refresh_info(self))
        self.all_functions = {
            dec: dec.capitalize() for dec in registry_mod.func_dict.keys()
        }
        self.scalar_functions = {
            k: v
            for (k, v) in self.all_functions.items()
            if k not in ("hide", "hist", "corr")
        }
        self.obs_flag = False
        super().__init__(children=[])

    def draw_matrix(self, ext_df=None):
        lst = [ipw.Label("")] + [ipw.Label(s) for s in self.all_functions.values()]
        width_ = len(lst)
        df = self.matrix_to_df() if ext_df is None else ext_df
        for col in sorted(self.visible_cols):
            _, col_type = col.split(":")
            lst.append(ipw.Label(col))
            for k in self.all_functions.keys():
                lst.append(self._info_checkbox(col, k, get_flag_status(col_type, k)))
        gb = ipw.GridBox(
            lst,
            layout=ipw.Layout(grid_template_columns=f"250px repeat({width_-1}, 50px)"),
        )
        if df is not None:
            for i in df.index:
                for c in df.columns:
                    self.info_cbx[(i, c)].value = bool(df.loc[i, c])
        return gb

    def lock_conf(self):
        self._hidden_sel_wg.disabled = True
        for cbx in self.info_cbx.values():
            cbx.disabled = True

    def unlock_conf(self):
        self._hidden_sel_wg.disabled = False
        for (key, func), cbx in self.info_cbx.items():
            _, dtype = key.split(":")
            cbx.disabled = get_flag_status(dtype, func)

    def matrix_to_df(self):
        if not self.info_cbx:
            return
        cols = self.visible_cols
        funcs = list(self.all_functions.keys())[1:]  # because 0 is "hide"
        arr = np.zeros((len(cols), len(funcs)), dtype=bool)
        for i, c in enumerate(self.visible_cols):
            for j, f in enumerate(funcs):
                arr[i, j] = self.info_cbx[(c, f)].value
        df = pd.DataFrame(arr, index=cols, columns=funcs)
        return df

    def make_btn_bar(self):
        self._btn_edit = make_button("Edit", disabled=False, cb=_make_btn_edit_cb(self))
        self._btn_cancel = make_button(
            "Cancel", disabled=True, cb=_make_btn_cancel_cb(self)
        )
        self._btn_apply = make_button(
            "Apply", disabled=True, cb=_make_btn_apply_cb(self)
        )
        self._btn_bar = ipw.HBox([self._btn_edit, self._btn_cancel, self._btn_apply])
        return self._btn_bar

    def get_histogram_widget(self, hname, hist_mod):
        if hname in self._hdict and self._hdict[hname][0] is hist_mod:
            return self._hdict[hname][1]
        name, type_ = hname.split(":")
        if type_ == "string":
            hout = VegaWidget(spec=bar_spec_no_data)
            bp_mod = hist_mod
            bp_mod.on_after_run(_refresh_info_barplot(hout, bp_mod))
        else:
            hmod_1d = hist_mod.histogram1d
            sk_mod = hist_mod.kll
            # bp_mod = hdict["barplot"]
            lower_mod = hist_mod.lower
            upper_mod = hist_mod.upper
            range_slider = bins_range_slider("Range:")
            range_widgets[name] = range_slider
            hout = ipw.VBox(
                [
                    ipw.VBox(
                        [
                            VegaWidget(spec=kll_spec_no_data),
                            ipw.HBox(
                                [
                                    range_slider,
                                    ipw.Label("Min:"),
                                    ipw.Label("Max:"),
                                ]
                            ),
                        ]
                    ),
                    VegaWidget(spec=hist1d_spec_no_data),
                ]
            )
            range_slider.observe(
                make_observer(name, sk_mod, lower_mod, upper_mod), "value"
            )
            sk_mod.on_after_run(_refresh_info_sketch(hout, sk_mod))
            hmod_1d.on_after_run(_refresh_info_hist_1d(hout, hmod_1d))
        self._hdict[hname] = [hist_mod, hout]
        return hout

    def get_selection_set(self, func):
        return set(self._last_df.index[self._last_df.loc[:, func] != 0])

    def refresh_info(self):
        print(".", end="")
        global debug_console
        if self._dshape_mod.result is None:
            return
        if not self.children:
            selm = ipw.SelectMultiple(
                options=self.hidden_cols,
                # options=[str(i) for i in range(100)],
                value=[],
                rows=5,
                description="Show",
                disabled=False,
            )
            self._hidden_sel_wg = selm
            self.visible_cols = [
                f"{k}:{t}" for (k, t) in self._dshape_mod.result.items()
            ]
            selm.observe(_make_selm_obs(self), "value")
            gb = self.draw_matrix()
            self.conf_box = ipw.VBox([selm, gb, self.make_btn_bar()])
            self.lock_conf()
            self.children = (self.conf_box,)
            self.set_next_title("Conf")
        if self._registry_mod._matrix is None:
            return
        mod_matrix = self._registry_mod._matrix

        if self.previous_visible_cols != self.visible_cols:
            ####
            lst = [ipw.Label("")] + [
                ipw.Label(s) for s in self.scalar_functions.values()
            ]
            width_ = len(lst)
            for col in self.visible_cols:
                lst.append(ipw.Label(col))
                for k in self.scalar_functions.keys():
                    lst.append(self._info_label((col, k)))
            gb = ipw.GridBox(
                lst,
                layout=ipw.Layout(
                    grid_template_columns=f"200px repeat({width_-1}, 120px)"
                ),
            )
            self.previous_visible_cols = self.visible_cols[:]
            # self.children += (gb,)
            # self.set_next_title("Main")
            self.set_tab("Main", gb)
        # refresh Main
        for col in self.visible_cols:
            col_name = col.split(":")[0]
            for k in self.scalar_functions.keys():
                lab = self.info_labels[(col, k)]
                if not self.info_cbx[(col, k)].value:
                    lab.value = ""
                    continue
                subm = mod_matrix.loc[col, k]
                if not (subm and subm.result):
                    lab.value = "..."
                    continue
                res = subm.result.get(col_name, "")
                res = format_label(res)
                lab.value = res
        # histograms
        if self._last_df is not None and np.any(self._last_df.loc[:, "hist"]):
            if self._hist_tab is None:
                self._hist_tab = DynTab()
            self.set_tab("Histograms", self._hist_tab, overwrite=False)
            hist_sel = set(self._last_df.loc[:, "hist"])
            if hist_sel != self._hist_sel:
                self._hist_tab.children = tuple([])
                for attr, ck in zip(self._last_df.index, self._last_df.loc[:, "hist"]):
                    if not ck:
                        continue
                    hist_mod = mod_matrix.loc[attr, "hist"]
                    assert hist_mod
                    hwg = self.get_histogram_widget(attr, hist_mod)
                    self._hist_tab.set_tab(attr, hwg, overwrite=False)
                self._hist_sel = hist_sel
        else:
            self.remove_tab("Histograms")
            self._hist_sel = None
        # corr
        if self._last_df is not None and np.any(self._last_df.loc[:, "corr"]):
            if "corr" not in self._registry_mod._multi_col_modules:
                return
            corr_mod = self._registry_mod._multi_col_modules["corr"]
            corr_sel = corr_mod._columns[:]
            if corr_sel != self._corr_sel:
                corr_out = VegaWidget(spec=corr_spec_no_data)
                self.set_tab("Correlation", corr_out)
                corr_mod.on_after_run(_refresh_info_corr(corr_out, corr_mod, self))
                self._corr_sel = corr_sel.copy()

        else:
            self.remove_tab("Correlation")
            self._corr_sel = None

        if IS_DEBUG:
            all_titles = self.get_titles()
            if "Console" not in all_titles:
                outp = ipw.Output()
                self.set_tab("Console", outp)

        return

    def get_corr_sel(self):
        return [
            col.split(":")[0]
            for ((col, fnc), ck) in self.info_cbx.items()
            if fnc == "corr" and ck.value
        ]

    def _info_label(self, k):
        lab = ipw.Label()
        self.info_labels[k] = lab
        return lab

    def _info_checkbox(self, col, func, dis):
        cbx = ipw.Checkbox(value=False, description="", disabled=dis, indent=False)
        self.info_cbx[(col, func)] = cbx
        cbx.observe(_make_cbx_obs(self, col, func), "value")
        return cbx
