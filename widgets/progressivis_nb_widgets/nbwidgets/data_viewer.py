import time
from functools import singledispatch
from collections import Iterable, Sequence
from itertools import product
import ipywidgets as ipw
#import altair as alt
import numpy as np
import pandas as pd
from .utils import update_widget
from progressivis.core import asynchronize, aio
from IPython.display import display, clear_output
from vega.widget import VegaWidget
from ._hist1d_schema import hist1d_spec_no_data
from ._corr_schema import corr_spec_no_data
from ._bar_schema import bar_spec_no_data


def corr_as_vega_dataset(mod, columns=None):
    """
    
    """
    if columns is None:
        columns = mod._columns
    def _c(kx, ky):
        return mod.result[frozenset([kx, ky])]
    return [dict(corr=_c(kx,ky),
                 corr_label=f"{_c(kx,ky):.2f}",
                 var=kx,
                 var2=ky)
            for (kx, ky) in product(columns, columns)]

def categ_as_vega_dataset(categs):
    return [{'category': k, 'count': v} for
            (k, v) in categs.items()]


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


def refresh_info_histogram1d(hout, hmod): # older
    if hout.bar:
        categs = hmod._categorical
        if categs is None:
            return
        dataset = categ_as_vega_dataset(categs)
        hout.update('data', remove='true', insert=dataset)
        return
    if not hmod.result:
        return
    #spec_with_data = spec_no_data.copy()
    res = hmod.result.last().to_dict()
    hist = res['array']
    min_ = res['min']
    max_ = res['max']
    #bins = np.linspace(min_, max_, len(hist))
    source = pd.DataFrame({
        #'bins': bins,
        'nbins': range(len(hist)),
        'level': hist,
    })
    hout.update('data', remove='true', insert=source)

def refresh_info_hist(hout, hmod):
    if hout.bar:
        categs = hmod.result
        if categs is None:
            return
        dataset = categ_as_vega_dataset(categs)
        hout.update('data', remove='true', insert=dataset)
        return
    if not hmod.result:
        return
    #spec_with_data = spec_no_data.copy()
    res = hmod.result
    hist = res['pmf']
    min_ = res['min']
    max_ = res['max']
    #bins = np.linspace(min_, max_, len(hist))
    source = pd.DataFrame({
        #'bins': bins,
        'nbins': range(len(hist)),
        'level': hist,
    })
    hout.children[0].update('data', remove='true', insert=source)

def refresh_info_hist_1d(hout, h1d_mod):    
    if not h1d_mod.result:
        return
    #spec_with_data = spec_no_data.copy()
    res = h1d_mod.result.last().to_dict()
    hist = res['array']
    min_ = res['min']
    max_ = res['max']
    #bins = np.linspace(min_, max_, len(hist))
    source = pd.DataFrame({
        #'bins': bins,
        'nbins': range(len(hist)),
        'level': hist,
    })
    hout.children[1].update('data', remove='true', insert=source)


def _refresh_info_hist(hout, hmod):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(refresh_info_hist, hout, hmod)
    return _coro

def _refresh_info_hist_1d(hout, hmod):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(refresh_info_hist_1d, hout, hmod)
    return _coro

def refresh_info_corr(cout, cmod, main_wg):
    if not cmod.result:
        return
    cols = main_wg.get_corr_sel()
    dataset = corr_as_vega_dataset(cmod, cols)
    cout.update('data', remove='true', insert=dataset)


def _refresh_info_corr(cout, cmod, wg):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(refresh_info_corr, cout, cmod, wg)
    return _coro
    
type_op_mismatches = dict(string=set(['var', 'corr']))

def get_disabled_flag(dt, op):
    return op in type_op_mismatches.get(dt, set())


class DataViewer(ipw.Tab):
    #info_keys = {'min': 'Min', 'max': 'Max'}
    def __init__(self, module, scheduler=None):
        self._module = module
        self.info_labels = {}
        self.info_cb = {}
        self._hdict = {}
        self._hist_tab = None
        self._hist_sel = set()
        self._module.after_run_proc = _refresh_info(self)
        self.info_keys = {dec: dec.capitalize() for dec in module.decorations}
        self.info_keys_ext = dict(hist='Hist', corr='Corr', **self.info_keys)
        super().__init__(children=[])

    def set_next_title(self, name):
        pos = len(self.children)-1
        self.set_title(pos, name)
        
    def refresh_info(self):
        if not self._module.visible_cols:
            return
        if not self.children:
            lst = [ipw.Label("")]+[ipw.Label(s) for s in self.info_keys_ext.values()]
            width_ = len(lst)
            dshape = self._module.dshape
            #dshape = {}
            for col in self._module.visible_cols:
                col_type = str(dshape.get(col, 'X'))
                lst.append(ipw.Label(f"{col}: {col_type}"))
                #lst.append(ipw.Label(col))
                for k in self.info_keys_ext.keys():
                    lst.append(self._info_checkbox((col, k), get_disabled_flag(col_type, k)))
            gb = ipw.GridBox(lst,
                             layout=ipw.Layout(
                                 grid_template_columns=f"250px repeat({width_-1}, 50px)"))
            self.children = (gb,)
            self.set_next_title('Conf')
            ####
            lst = [ipw.Label("")]+[ipw.Label(s) for s in self.info_keys.values()]
            width_ = len(lst)
            for col in self._module.visible_cols:
                lst.append(ipw.Label(col))
                for k in self.info_keys.keys():
                    lst.append(self._info_label((col, k)))
            gb = ipw.GridBox(lst,
                             layout=ipw.Layout(
                                 grid_template_columns=f"repeat({width_}, 120px)"))
            self.children += (gb,)
            self.set_next_title('Main')
            
            if hasattr(self._module, 'hist'):
                for hname, hmod in self._module.hist.items():
                    if str(dshape.get(hmod.column)) == 'string':
                        hout = VegaWidget(spec=bar_spec_no_data)
                        hout.bar = True
                    else:
                        hout = ipw.HBox([VegaWidget(spec=hist1d_spec_no_data),
                                         VegaWidget(spec=hist1d_spec_no_data)])
                        hout.bar = False
                    self._hdict[hname] = hout
                    hmod.after_run_proc =  _refresh_info_hist(hout, hmod)
                    if not hout.bar:
                        hmod_1d = self._module.hist1d[hname]
                        hmod_1d.after_run_proc =  _refresh_info_hist_1d(hout, hmod_1d)
                self._hist_tab = ipw.Tab()
                #self._hist_tab.children = tuple(self._hdict.values())
                #for i, k in enumerate(self._module.hist.keys()):
                #    self._hist_tab.set_title(i+1, k)
                self.children += (self._hist_tab,)
                self.set_next_title("1D Hist")
            if hasattr(self._module, 'corr'):
                corr_out = VegaWidget(spec=corr_spec_no_data)
                corr_mod = self._module.corr
                corr_mod.after_run_proc =  _refresh_info_corr(corr_out, corr_mod, self)
                self.children += (corr_out,)
                self.set_next_title("Corr")
            #self._debug = ipw.Output()
            #self.children += (self._debug,)
            #self.set_next_title("Debug")
        # refresh Main
        for col in self._module.visible_cols:
            for k in self.info_keys.keys():
                if not self.info_cb[(col, k)].value:
                    continue
                lab = self.info_labels[(col, k)]
                subm = getattr(self._module, k)
                res = subm.result.get(col,"")
                res = format_label(res)
                lab.value = res
        # refresh hist
        hist_sel = set([col for ((col, fnc),ck) in self.info_cb.items() if fnc=='hist' and ck.value])
        if hist_sel != self._hist_sel:
            #with self._debug:
            #    print("changes...")
            selection_ = [(k, v) for (k,v) in self._hdict.items() if k in hist_sel]
            selection_k, selection_v = zip(*selection_)
            self._hist_tab.children = tuple(list(selection_v))
            for i, k in enumerate(selection_k):
                    self._hist_tab.set_title(i, k)
            self._hist_sel = hist_sel

    def get_corr_sel(self):
        return [col for ((col, fnc),ck) in self.info_cb.items() if fnc=='corr' and ck.value]
            
    def _info_label(self, k):
        lab = ipw.Label()
        self.info_labels[k] = lab
        return lab

    def _info_checkbox(self, k, dis):
        cb = ipw.Checkbox(
            value=False,
            description='',
            disabled=dis,
            indent=False
        )
        self.info_cb[k] = cb
        return cb





