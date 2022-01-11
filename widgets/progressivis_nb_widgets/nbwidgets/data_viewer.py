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
range_widget = None


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
    """

    """
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
    cols = main_wg.get_corr_sel()
    dataset = corr_as_vega_dataset(cmod, cols)
    cout.update("data", remove="true", insert=dataset)


def _refresh_info_corr(cout, cmod, wg):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(refresh_info_corr, cout, cmod, wg)

    return _coro


type_op_mismatches = dict(string=set(["var", "corr"]))


def get_disabled_flag(dt, op):
    return op in type_op_mismatches.get(dt, set())


class DataViewer(ipw.Tab):
    # info_keys = {'min': 'Min', 'max': 'Max'}
    def __init__(self, module, scheduler=None):
        self._module = module
        self.info_labels = {}
        self.info_cb = {}
        self._hdict = {}
        self._hist_tab = None
        self._hist_sel = set()
        self._module.after_run_proc = _refresh_info(self)
        self.info_keys = {dec: dec.capitalize() for dec in module.decorations}
        self.info_keys_ext = dict(hist="Hist", corr="Corr", **self.info_keys)
        super().__init__(children=[])

    def set_next_title(self, name):
        pos = len(self.children) - 1
        self.set_title(pos, name)

    def refresh_info(self):
        global debug_console, range_widget
        if not self._module.visible_cols:
            return
        if not self.children:
            lst = [ipw.Label("")] + [ipw.Label(s) for s in self.info_keys_ext.values()]
            width_ = len(lst)
            dshape = self._module.dshape
            # dshape = {}
            for col in self._module.visible_cols:
                col_type = str(dshape.get(col, "X"))
                lst.append(ipw.Label(f"{col}: {col_type}"))
                # lst.append(ipw.Label(col))
                for k in self.info_keys_ext.keys():
                    lst.append(
                        self._info_checkbox((col, k), get_disabled_flag(col_type, k))
                    )
            gb = ipw.GridBox(
                lst,
                layout=ipw.Layout(
                    grid_template_columns=f"250px repeat({width_-1}, 50px)"
                ),
            )
            self.children = (gb,)
            self.set_next_title("Conf")
            ####
            lst = [ipw.Label("")] + [ipw.Label(s) for s in self.info_keys.values()]
            width_ = len(lst)
            for col in self._module.visible_cols:
                lst.append(ipw.Label(col))
                for k in self.info_keys.keys():
                    lst.append(self._info_label((col, k)))
            gb = ipw.GridBox(
                lst, layout=ipw.Layout(grid_template_columns=f"repeat({width_}, 120px)")
            )
            self.children += (gb,)
            self.set_next_title("Main")
            if hasattr(self._module, "hist"):
                for hname, hdict in self._module.hist.items():
                    hmod_1d = hdict["hist1d"]
                    sk_mod = hdict["sketching"]
                    bp_mod = hdict["barplot"]
                    lower_mod = hdict["lower"]
                    upper_mod = hdict["upper"]
                    if str(dshape.get(sk_mod.column)) == "string":
                        hout = VegaWidget(spec=bar_spec_no_data)
                        is_string = True
                    else:
                        range_slider = bins_range_slider("Range:")
                        range_widget = range_slider
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

                        range_slider.observe(_observe_range, "value")

                        is_string = False
                    self._hdict[hname] = hout
                    if is_string:
                        bp_mod.after_run_proc = _refresh_info_barplot(hout, bp_mod)
                    else:
                        sk_mod.after_run_proc = _refresh_info_sketch(hout, sk_mod)
                        hmod_1d.after_run_proc = _refresh_info_hist_1d(hout, hmod_1d)
                self._hist_tab = ipw.Tab()
                self.children += (self._hist_tab,)
                self.set_next_title("1D Hist")
            if hasattr(self._module, "corr"):
                corr_out = VegaWidget(spec=corr_spec_no_data)
                corr_mod = self._module.corr
                corr_mod.after_run_proc = _refresh_info_corr(corr_out, corr_mod, self)
                self.children += (corr_out,)
                self.set_next_title("Corr")
            if IS_DEBUG:
                outp = ipw.Output()
                debug_console = outp
                self.children += (outp,)
                self.set_next_title("Console")
        # refresh Main
        for col in self._module.visible_cols:
            for k in self.info_keys.keys():
                if not self.info_cb[(col, k)].value:
                    continue
                lab = self.info_labels[(col, k)]
                subm = getattr(self._module, k)
                res = subm.result.get(col, "")
                res = format_label(res)
                lab.value = res
        # refresh hist
        hist_sel = set(
            [
                col
                for ((col, fnc), ck) in self.info_cb.items()
                if fnc == "hist" and ck.value
            ]
        )
        if hist_sel != self._hist_sel:
            selection_ = [(k, v) for (k, v) in self._hdict.items() if k in hist_sel]
            if selection_:
                selection_k, selection_v = zip(*selection_)
                self._hist_tab.children = tuple(list(selection_v))
                for i, k in enumerate(selection_k):
                    self._hist_tab.set_title(i, k)
            else:
                self._hist_tab.children = tuple()
            self._hist_sel = hist_sel

    def get_corr_sel(self):
        return [
            col
            for ((col, fnc), ck) in self.info_cb.items()
            if fnc == "corr" and ck.value
        ]

    def _info_label(self, k):
        lab = ipw.Label()
        self.info_labels[k] = lab
        return lab

    def _info_checkbox(self, k, dis):
        cb = ipw.Checkbox(value=False, description="", disabled=dis, indent=False)
        self.info_cb[k] = cb
        return cb
