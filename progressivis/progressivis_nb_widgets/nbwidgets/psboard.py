import ipywidgets as ipw
import ipysheet as ips
import numpy as np
from .control_panel import ControlPanel
from .sensitive_html import SensitiveHTML
from .utils import *
from .module_graph import ModuleGraph
from progressivis.core import JSONEncoderNp
from io import StringIO
from jinja2 import Template
from .templates import *
from .module_wg import *
import json

commons = {}
debug_console = ipw.Output()
#
# Coroutines
#

async def module_choice(psboard):
    while True:
        await wait_for_change(psboard.htable, 'value')
        #with debug_console:
        #    print("Clicked: ", psboard.htable.value)
        if len(psboard.tab.children)<3:
            psboard.tab.children = psboard.tab.children + (psboard.current_module,)
        psboard.current_module.module_name =  psboard.htable.value[len(psboard.htable.sensitive_css_class)+1:]
        psboard.tab.set_title(2, psboard.current_module.module_name)
        psboard.tab.selected_index = 2
        psboard.refresh()

async def change_tab(psboard):
    while True:
        await wait_for_change(psboard.tab, 'selected_index')
        with debug_console:
            print("Changed: ", psboard.tab.selected_index)        
        psboard.refresh()
        
async def refresh_(psboard):
    if psboard.refresh_event is None:
        psboard.refresh_event = aio.Event()
    while True:
        await psboard.refresh_event.wait()
        psboard.refresh_event.clear()
        psboard.refresh()
        await aio.sleep(0.5)

async def control_panel(psboard, action):
    btn, cb =  psboard.cpanel.cb_args(action)
    while True:
        await wait_for_click(btn, cb)

# end coros

class PsBoard(ipw.VBox):
    def __init__(self, scheduler=None):
        global debug_console
        self.scheduler = scheduler
        self._cache = None
        self._cache_js = None
        self.cpanel = ControlPanel(scheduler)
        #self.current_module_name = None
        #self.current_module = SensitiveHTML()
        self.current_module = ModuleWg(self, debug_console) 
        #self.current_module.sensitive_css_class = 'foo'
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
        self.htable = SensitiveHTML()
        self.refresh_event = None
        commons.update(tab=self.tab, scheduler=self.scheduler)
        #self.init_sheet()
        #self.make_table()
        #self.tab.children = [self.sheet, self.mgraph]
        super().__init__([self.cpanel, self.tab, debug_console])

    def make_table_index(self, modules):
        tmpl = Template(index_tpl)
        self.htable.sensitive_css_class = 'ps-row-btn'
        self.htable.data = tmpl.render(modules=modules, cols=self.cols)
    

    def enable_refresh(self):
        json_ = self.scheduler.to_json(short=False)
        self._cache = JSONEncoderNp.dumps(json_)
        self._cache_js = None
        if self.refresh_event is None:
            self.refresh_event = aio.Event()
        self.refresh_event.set()

    @property
    def coroutines(self):
        return [module_choice(self), refresh_(self),
                control_panel(self, "resume"),
                control_panel(self, "stop"), 
                control_panel(self, "step"), change_tab(self)]
    
    def refresh(self):
        if self._cache is None:
            return
        if self._cache_js is None:
            self._cache_js = JSONEncoderNp.loads(self._cache)
        json_ = self._cache_js
        self.cpanel.run_nb.value = str(json_['run_number'])
        if self.tab.selected_index == 0:
            self.make_table_index(json_['modules'])
        elif self.tab.selected_index == 1:
            self.mgraph.data = self._cache
        else:
            assert len(self.tab.children)>2
            self.current_module.refresh()
            
            """for i, m in enumerate(json_['modules']):
                if m['id']==self.current_module_name:
                    self.tab.children[2].data = layout_dict(m, order=["classname",
                                   "speed",
                                   "output_slots",
                                   "debug",
                                   "state",
                                   "last_update",
                                   "default_step_size",
                                   "start_time",
                                   "end_time",
                                   "parameters",
                                   "input_slots"])
                    break
            else:
                raise ValueError(f"Unknown module {self.current_module_name}")
            """
        if len(self.tab.children)<3:
            self.tab.children = [self.htable, self.mgraph]
        else:
            third =  self.tab.children[2]
            self.tab.children = [self.htable, self.mgraph, third]
