import time
import ipywidgets as ipw
#import altair as alt
import numpy as np
import pandas as pd
from .utils import update_widget
from progressivis.core import asynchronize, aio
from IPython.display import display, clear_output
#from altair_saver import save
#import matplotlib.pyplot as plt
from vega.widget import VegaWidget

# "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
"""
"selection": {"pts": {"type": "single", "encodings": ["x"]}},

              "axis": {
                  "labelAngle": 0,
                  "labelOverlap": "parity"
              }
"""
spec_no_data = {
    "height": 500,
    "width": 500,
    "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
    "mark": "bar",
    "data": {
        "name": "data"
    },
    "encoding": {
        "x": {"type": "quantitative", "scale": {"domain": [0,127]},
              "field": "bucket",
        },
        "y": {"type": "quantitative", "field": "level"},

    }
}



def _refresh_info(wg):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(wg.refresh_info)
    return _coro


def mpl_refresh_info_hist(hout, hmod):
    #import pdb;pdb.set_trace()
    #pass
    if not hmod.result:
        return
    #with hout:
    #print(hmod.result.last().to_dict())
    #plt.clear()
    hist = hmod.result.last().to_dict()['array']
    bins = np.array(range(len(hist)+1))
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots()#figsize=(8,3))
    ax.bar(center, hist, align='center', width=width)
    ax.set_xticks(bins)
    with hout:
        plt.show(fig)
    #from io import BytesIO
    #f = BytesIO()
    #plt.savefig(f, format="png")
    #wg = ipw.Image(value=f.getvalue(), width=512, height=512)
    #with hout:
    #    display(wg)
    #hout.value=f.getvalue()
    #print("width:", width)
    #print("hist:", len(hist), hist)
    #print("center:", len(center),center)

def altair_refresh_info_hist(hout, hmod):
    if not hmod.result:
        return
    hist = hmod.result.last().to_dict()['array']
    bins = np.array(range(len(hist)+1))
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    source = pd.DataFrame({
        'a': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
        'b': [28, 55, 43, 91, 81, 53, 19, 87, 52]
    })

    c = alt.Chart(source).mark_bar().encode(
        x='a',
        y='b'
    )
    from io import BytesIO
    f = BytesIO()
    res = save(c, f, fmt='png')
    assert res is None
    assert isinstance(f.getvalue(), bytes)
    hout.value=f.getvalue()
    #with hout:
    #    clear_output()
    #    c.display()

def refresh_info_hist(hout, hmod):
    #import pdb;pdb.set_trace()
    #pass
    if not hmod.result:
        return
    spec_with_data = spec_no_data.copy()
    res = hmod.result.last().to_dict()
    hist = res['array']
    min_ = res['min']
    max_ = res['max']
    ix = np.array(range(len(hist)))
    #h = 10**(int(np.log10(np.max(hist)))+1)
    bins = np.linspace(min_, max_, len(hist))
    for i in [0,1]: # avoids blink
        half_h = hist[i::2]
        half_b = bins[i::2]
        half_ix = ix[i::2]
        remove_expr = '||'.join([f"datum.ix=={j}" for j in half_ix])
        source = pd.DataFrame({
            'bucket': half_ix,
            'level': half_h,
            #'ix': half_ix
        })
        spec_with_data["data"] = {
            "name": "data",
            "values": source.to_dict(orient='records'),
        }
        val = source.to_dict(orient='records')
        hout.update('data', remove=remove_expr)
        hout.update('data', insert=val)

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


class IScalerOut(ipw.Tab):
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
        #super().__init__(lst, layout=ipw.Layout(grid_template_columns="repeat(2, 120px)"))
        gb = ipw.GridBox(lst, layout=ipw.Layout(grid_template_columns="repeat(2, 120px)"))
        if not module.hist:
            super().__init__(children=[gb])
            self.set_title(0, 'ScalerMinMax')
        else:
            hlist = [] #[ipw.Output() for _ in range(len(module.hist))]
            for hmod in module.hist.values():
                hout = VegaWidget(spec=spec_no_data) #ipw.Output() #ipw.Image(value=b'', width=512, height=512)
                hlist.append(hout)
                hmod.after_run_proc =  _refresh_info_hist(hout, hmod)
            super().__init__(children=[gb, ipw.VBox(hlist)])
            self.set_title(0, 'ScalerMinMax')
            self.set_title(1, 'Histograms')




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
