import time
import ipywidgets as ipw
#import altair as alt
import numpy as np
import pandas as pd
from .utils import update_widget
from progressivis.core import asynchronize, aio
from IPython.display import display, clear_output
from vega.widget import VegaWidget

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

def _refresh_info(wg):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(wg.refresh_info)
    return _coro



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



class IScalerIn(ipw.GridBox):
    def __init__(self, module, scheduler=None):
        self._module = module
        self.info_labels = {}
        rt = ipw.IntSlider(
            value=1000,
            min=0,
            max=100_000,
            step=1000,
            description='Reset threshold:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        tol = ipw.IntSlider(
            value=5,
            min=0,
            max=100,
            step=1,
            description='Tolerance (%):',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        ign = ipw.IntSlider(
            value=10,
            min=0,
            max=1000,
            step=1,
            description='Ignore max:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        btn = ipw.Button(
            description='Apply',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Apply',
            icon='check' # (FontAwesome names without the `fa-` prefix)
        )
        btn.on_click(self.get_apply_cb())
        self._apply = btn
        rst = ipw.Checkbox(
            value=False,
            description='Reset:',
            disabled=False,
            indent=False
        )
        self._dict = {'reset_threshold': rt, 'delta': tol, 'ignore_max': ign, 'reset': rst}
        lst = []
        for wg in [rt, tol, ign, rst]:
            lst.append(ipw.Label(wg.description))
            wg.description = ''
            lst.append(wg)
        super().__init__(lst+[btn], layout=ipw.Layout(grid_template_columns="repeat(2, 120px)"))

    @property
    def values(self):
        return {k: wg.value for (k, wg) in self._dict.items()}

    def get_apply_cb(self):
        def _cbk(_btn):
            _ = _btn
            m = self._module
            values = dict(self.values) # shallow copy
            values['time'] = time.time() # always make a change
            #wg._dict['reset'].value = False
            loop = aio.get_event_loop()
            loop.create_task(m.control.from_input(values))
        return _cbk


class IScalerOut(ipw.HBox):
    info_keys = {'clipped': 'Clipped:', 'ignored': 'Ignored:',
                 'needs_changes': 'Needs changes:', 'has_buffered': 'Has buff:', 'last_reset': 'Last reset:'}
    def __init__(self, module, scheduler=None):
        self._module = module
        self.info_labels = {}
        lst = []
        for k, lab in self.info_keys.items():
            lst.append(ipw.Label(lab))
            lst.append(self._info_label(k))
        lst.append(ipw.Label('Rows'))
        self._rows_label = ipw.Label('0')
        lst.append(self._rows_label)
        self._module.after_run_proc = _refresh_info(self)
        gb = ipw.GridBox(lst, layout=ipw.Layout(grid_template_columns="repeat(2, 120px)"))
        if not module.hist:
            super().__init__(children=[gb])
        else:
            hlist = []
            for hmod in module.hist.values():
                hout = VegaWidget(spec=spec_no_data)
                hlist.append(hout)
                hmod.after_run_proc =  _refresh_info_hist(hout, hmod)
            htab = ipw.Tab(hlist)
            for i, t in enumerate(module.hist.keys()):
                htab.set_title(i, t)
            super().__init__(children=[gb, htab])




    def _info_label(self, k):
        v = ''
        if self._module._info:
            v = str(self._module._info.get(k, ''))
        lab = ipw.Label(v)
        self.info_labels[k] = lab
        return lab

    def refresh_info(self):
        if not self._module._info:
            return
        for k, v in self._module._info.items():
            lab = self.info_labels[k]
            lab.value = str(v)
        if self._module.result is None:
            return
        self._rows_label.value = str(len(self._module.result))
