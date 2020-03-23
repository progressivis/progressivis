import ipywidgets as ipw
import ipysheet as ips
import numpy as np
from .control_panel import ControlPanel
from .utils import *
from .module_graph import ModuleGraph
from progressivis.core import JSONEncoderNp


commons = {}
console = ipw.Output()

async def _btn_cb(btn, cb):
    while True:
        await wait_for_click(btn, cb)


def _on_button_clicked2(b):
    #import pdb;pdb.set_trace()
    tab = commons['tab']
    if len(tab.children)==2:
        nchildren = tab.children+(ipw.Output(),)
        tab.children = nchildren
    tab.set_title(2, b.description)
    with tab.children[2]:
        print(b.description)
def _on_button_clicked(b):
    with console:
        print("Clicked!")
    try:
        _on_button_clicked2(b)
    except Exception as e:
        with console:
            print(e)
def _btn(label):
    btn = ipw.Button(
        description=label,
        disabled=False,
        #button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltip=label,
        layout=ipw.Layout(flex='1 1 auto', width='auto')
        #style={'textAlign': 'left'}
        #icon='check' # (FontAwesome names without the `fa-` prefix)
    )
    #btn.on_click(_on_button_clicked)
    return btn



class PsBoard(ipw.VBox):
    def __init__(self, scheduler=None):
        global console
        self.scheduler = scheduler
        self.cpanel = ControlPanel(scheduler)
        """
        modules = scheduler.to_json()['modules']
        titles = ['Id', 'Class', 'State', 'Last Update', 'Order']
        cols = ['id', 'classname', 'state', 'last_update', 'order']
        col_width = [100, 100, 50, 40, 40]
        self.sheet = ips.sheet(rows=len(modules), columns=len(cols),
                               column_headers=titles, column_width=col_width)
        #self.sheet.column_width = col_width
        for i, m in enumerate(modules):
            for j, c in enumerate(cols):
                ips.cell(i, j, str(m[c]), read_only=True)
        """
        self.mgraph = ModuleGraph()
        self.tab = ipw.Tab()
        #self.tab.children = [self.sheet, gr]
        self.tab.set_title(0, 'Modules')
        self.tab.set_title(1, 'Module graph')
        self.state = []
        self.last_update = []
        self.btns = []
        self.msize = 0
        self.cols = ['id', 'classname', 'state', 'last_update', 'order']
        commons.update(tab=self.tab, scheduler=self.scheduler)
        self.init_sheet()
        #self.tab.children = [self.sheet, self.mgraph]
        super().__init__([self.cpanel, self.tab, console])

    def init_sheet(self, modules=None):
        try:
            self.sheet.close()
        except:
            pass
        self.state = []
        self.last_update = []
        if modules is None:
            modules = self.scheduler.to_json()['modules']
        self.msize = len(modules)
        titles = ['Id', 'Class', 'State', 'Last Update', 'Order']
        #cols = ['id', 'classname', 'state', 'last_update', 'order']
        col_width = [100, 100, 50, 40, 40]
        self.sheet = ips.sheet(rows=len(modules), columns=len(self.cols),
                               column_headers=titles, column_width=col_width)
        #self.sheet.column_width = col_width
        for i, m in enumerate(modules):
            for j, c in enumerate(self.cols):
                if c=='id':
                    b_ =  _btn(m[c])
                    cell = ips.cell(i, j,b_)
                    self.btns.append(b_)
                else:
                    cell = ips.cell(i, j, str(m[c]), read_only=True)
                if c=='state':
                    self.state.append(cell)
                elif c=='last_update':
                    self.last_update.append(cell)
    @property
    def btn_cb(self):
        return [(b, _on_button_clicked) for b in self.btns]
    @property
    def btn_coros(self):
        return [_btn_cb(b,cb) for b, cb in self.btn_cb]
    def refresh_modules(self, json_):
        modules = json_['modules']
        if len(modules)!=self.msize:
            self.init_sheet(modules)
        for i, m in enumerate(modules):
            for j, c in enumerate(self.cols):
                if c=='state' and self.state[i].value != m[c]:
                    self.state[i].value = m[c]
                elif c=='last_update':
                    self.last_update[i].value = str(m[c])
                    self.last_update[i].send_state()
                    #with console:
                    #    print(m[c])
        self.sheet.send_state()
    def refresh(self):
        json_ = self.scheduler.to_json(short=True)
        self.refresh_modules(json_)
        #self.mgraph.data = JSONEncoderNp.dumps(json_)
        if len(self.tab.children)<3:
            self.tab.children = [self.sheet, self.mgraph]
        else:
            third =  self.tab.children[2]
            self.tab.children = [self.sheet, self.mgraph, third]
            """        
tab.children = [sc, gr]
tab.set_title(0, 'Modules')
tab.set_title(1, 'Module graph')
vbox = ipw.VBox([tab, cpanel])
vbox
"""
