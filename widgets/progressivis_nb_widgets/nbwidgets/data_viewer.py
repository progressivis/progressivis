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
from ._corr_schema import corr_spec_no_data

spec_no_data = {
    "data": {
        "name": "data"
    },
    "height": 500,
    "width": 500,
    "layer": [
        {
            "mark": "bar",
            "$schema": "https://vega.github.io/schema/vega-lite/v4.8.1.json",
            "encoding": {
                "x": {"type": "ordinal",
                      "field": "nbins",
                      #"title": "Values", #"axis": {"format": ".2e", "ticks": False},
                      "title": "Values",
                      "axis": {"format": ".2e", "labelExpr": "(datum.value%10>0 ? null : datum.value)"},
                      #"axis": {"labelExpr": "datum.label"},
                },
                "y": {"type": "quantitative", "field": "level", "title": "Count"},

            }
        },
        {
            "mark": "rule",
            "encoding": {
                "x": {"aggregate": "min", "field": "bins", "title": None, "axis": {"tickCount": 0}},
                "color": {"value": "red"},
                "size": {"value": 1}
            }
        },
        {
            "mark": "rule",
            "encoding": {
                "x": {"aggregate": "max", "field": "bins", "title": None, "axis": {"tickCount": 0}},
                "color": {"value": "red"},
                "size": {"value": 1}
            }
        }

    ]
}

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


def refresh_info_hist(hout, hmod):
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
    #spec_with_data["data"] = {
    #    "name": "data",
    #    "values": source.to_dict(orient='records'),
    #}
    #hout.spec = spec_with_data
    hout.update('data', remove='true', insert=source)


def _refresh_info_hist(hout, hmod):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(refresh_info_hist, hout, hmod)
    return _coro

def refresh_info_corr(cout, cmod, wg):
    if not cmod.result:
        return
    cols = wg.get_corr_sel()
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
                    hout = VegaWidget(spec=spec_no_data)
                    self._hdict[hname] = hout
                    hmod.after_run_proc =  _refresh_info_hist(hout, hmod)
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
            self._hist_tab.children = tuple([v for (k,v) in self._hdict.items() if k in hist_sel])
            for i, k in enumerate(hist_sel):
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





