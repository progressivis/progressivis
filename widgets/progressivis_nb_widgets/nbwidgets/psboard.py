from collections import defaultdict
import ipywidgets as ipw
from jinja2 import Template
from progressivis.core import JSONEncoderNp
import progressivis.core.aio as aio
from .control_panel import ControlPanel
from .sensitive_html import SensitiveHTML
from .utils import wait_for_change, wait_for_click, update_widget
from .module_graph import ModuleGraph
from .module_wg import ModuleWg


commons = {}
debug_console = ipw.Output()
#
# Coroutines
#

INDEX_TEMPLATE = """
<table id="mysortedtable" class="table table-striped table-bordered table-hover table-condensed">
<thead><tr><th></th><th>Id</th><th>Class</th><th>State</th><th>Last Update</th><th>Order</th></tr></thead>
<tbody>
{% for m in modules%}
  <tr>
  {% for c in cols%}
  <td>
  {% if c=='id' %}
  <a class='ps-row-btn' id="ps-row-btn_{{m[c]}}" type='button' >{{m[c]}}</a>
  {% elif c=='is_visualization' %}
  <span id="ps-cell_{{m['id']}}_{{c}}">{{'a' if m[c] else ' '}}</span>
  {% else %}
  <span id="ps-cell_{{m['id']}}_{{c}}">{{m[c]}}</span>
  {% endif %}
  </td>
  {%endfor %}
  </tr>
{%endfor %}
</tbody>
</table>
"""


async def module_choice(psboard):
    while True:
        await wait_for_change(psboard.htable, 'value')
        # with debug_console:
        #    print("Clicked: ", psboard.htable.value)
        if len(psboard.tab.children) < 3:
            psboard.tab.children += (psboard.current_module,)
        psboard.current_module.module_name = psboard.htable.value[
            len(psboard.htable.sensitive_css_class)+1:]
        psboard.current_module.selection_changed = True
        psboard.tab.set_title(2, psboard.current_module.module_name)
        psboard.tab.selected_index = 2
        # await psboard.refresh()

# async def change_tab(psboard):
#     while True:
#         await wait_for_change(psboard.tab, 'selected_index')
#         with debug_console:
#             print("Changed: ", psboard.tab.selected_index)
#         psboard.refresh()


async def refresh_fun(psboard):
    while True:
        # await psboard.refresh_event.wait()
        # psboard.refresh_event.clear()
        json_ = psboard.scheduler.to_json(short=False)
        # pylint: disable=protected-access
        psboard._cache = JSONEncoderNp.dumps(json_, skipkeys=True)
        psboard._cache_js = None
        await psboard.refresh()
        await aio.sleep(0.5)


async def control_panel(psboard, action):
    btn, cbk = psboard.cpanel.cb_args(action)
    while True:
        await wait_for_click(btn, cbk)

# end coros


class PsBoard(ipw.VBox):  # pylint: disable=too-many-ancestors,too-many-instance-attributes
    def __init__(self, scheduler=None):
        global debug_console  # pylint: disable=global-statement
        self.scheduler = scheduler
        self._cache = None
        self._cache_js = None
        self.cpanel = ControlPanel(scheduler)
        self.current_module = ModuleWg(self, debug_console)
        self.mgraph = ModuleGraph()
        self.tab = ipw.Tab()
        self.tab.set_title(0, 'Modules')
        self.tab.set_title(1, 'Module graph')
        self.state = []
        self.last_update = []
        self.btns = []
        self.msize = 0
        self.cols = ['is_visualization', 'id', 'classname', 'state',
                     'last_update', 'order']
        self.htable = SensitiveHTML(layout=ipw.Layout(height='500px',
                                                      overflow='auto'))
        self.refresh_event = None
        self.other_coros = []
        self.vis_register = defaultdict(list)
        commons.update(tab=self.tab, scheduler=self.scheduler)
        super().__init__([self.cpanel, self.tab, debug_console])

    async def make_table_index(self, modules):
        if not self.htable.html:
            tmpl = Template(INDEX_TEMPLATE)
            await update_widget(self.htable,
                                'sensitive_css_class', 'ps-row-btn')
            await update_widget(self.htable,
                                'sort_table_ids', ['mysortedtable'])
            html = tmpl.render(modules=modules, cols=self.cols)
            # print(html)
            await update_widget(self.htable,
                                'html', html)
        else:
            data = {}
            for m in modules:
                for c in self.cols:
                    dataid = f"ps-cell_{m['id']}_{c}"
                    if c == 'is_visualization':
                        # Show an Unicode eye next to visualizations
                        data[dataid] = '\U0001F441' if m[c] else ' '
                    else:
                        data[dataid] = m[c]
            await update_widget(self.htable, 'data', data)

    def register_visualisation(self, widget, module,
                               label="Visualisation", glue=None):
        """
        called from notebook

        if module_class is None and module_id is None:
            raise ValueError("One and only one of 'module_class' and 'module_id' args must be defined")
        if not(module_class is None or module_id is None):
            raise ValueError("One and only one of 'module_class' and 'module_id' args must be defined")
        """
        linkable = hasattr(widget, 'link_module')
        if not linkable and glue is None:
            raise ValueError("Registering a visualisation requires a linkable "
                             "widget (i.e. which implements the "
                             "'link_module' interface) or 'glue' arg to be "
                             "provides with a valid 'glue' function")
        if glue is not None:
            self.other_coros += glue(widget, module)
        else:
            self.other_coros += widget.link_module(module, refresh=False)
        self.vis_register[module.name].append((widget, label))

    @property
    def coroutines(self):
        return [refresh_fun(self), module_choice(self),
                control_panel(self, "resume"),
                control_panel(self, "stop"),
                control_panel(self, "step")]+self.other_coros

    async def refresh(self):
        if self._cache is None:
            return
        if self._cache_js is None:
            self._cache_js = JSONEncoderNp.loads(self._cache)
        json_ = self._cache_js
        # self.cpanel.run_nb.value = str(json_['run_number'])
        await update_widget(self.cpanel.run_nb, 'value',
                            str(json_['run_number']))
        if self.tab.selected_index == 0:
            await self.make_table_index(json_['modules'])
        elif self.tab.selected_index == 1:
            # self.mgraph.data = self._cache
            await update_widget(self.mgraph, 'data', self._cache)
        else:
            assert len(self.tab.children) > 2
            await self.current_module.refresh()
        if len(self.tab.children) < 3:
            self.tab.children = [self.htable, self.mgraph]
        else:
            third = self.tab.children[2]
            self.tab.children = [self.htable, self.mgraph, third]
