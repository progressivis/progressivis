import time
import ipywidgets as ipw
from .utils import wait_for_change, wait_for_click, update_widget
from progressivis.core import aio

async def _apply(wg):
    def _cbk():
        m = wg._module
        values = dict(wg.values) # shallow copy
        values['time'] = time.time() # always make a change
        wg._dict['reset'].value = False
        aio.create_task(m.control.from_input(values))
    while True:
        await wait_for_click(wg._apply, _cbk)

async def _refresh_info(wg):
    while True:
        wg.refresh_info()
        await aio.sleep(0.5)

class IScaler(ipw.GridBox):
    info_keys = {'clipped': 'Clipped:', 'ignored': 'Ignored:', 'needs_changes': 'Needs changes:'}    
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
        for k, lab in self.info_keys.items():
            lst.append(ipw.Label(lab))
            lst.append(self._info_label(k))
        lst.append(ipw.Label('Rows'))
        self._rows_label = ipw.Label('0')
        lst.append(self._rows_label)
        super().__init__(lst+[btn], layout=ipw.Layout(grid_template_columns="repeat(2, 120px)"))
    @property
    def values(self):
        return {k: wg.value for (k, wg) in self._dict.items()}
    @property
    def coroutines(self):
        return [ _apply(self), _refresh_info(self)]

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
