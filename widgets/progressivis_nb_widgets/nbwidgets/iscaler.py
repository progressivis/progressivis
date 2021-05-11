import time
import ipywidgets as ipw
from .utils import update_widget
from progressivis.core import asynchronize, aio
        
def _refresh_info(wg):
    async def _coro(_1, _2):
        _ = _1, _2
        await asynchronize(wg.refresh_info)
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

        
class IScalerOut(ipw.GridBox):
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
        super().__init__(lst, layout=ipw.Layout(grid_template_columns="repeat(2, 120px)"))

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
