from __future__ import annotations

from collections import defaultdict
import ipywidgets as ipw  # type: ignore
from jinja2 import Template
from progressivis.core import JSONEncoderNp
from .control_panel import ControlPanel
from .sensitive_html import SensitiveHTML
from .utils import update_widget
from .module_graph import ModuleGraph
from .module_wg import ModuleWg

from typing import (
    Any,
    Optional,
    Literal,
    Callable,
    List,
    Dict,
    cast,
    Iterable,
    Coroutine,
    Set,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from progressivis.core.scheduler import Scheduler
    from progressivis.core.module import Module, JSon

WidgetType = Any

debug_console = ipw.Output()

INDEX_TEMPLATE = """
<table class="table table-striped table-bordered table-hover table-condensed">
<thead><tr><th></th><th>Id</th><th>Class</th><th>State</th><th>Last Update</th>
<th>Order</th></tr></thead>
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


def module_choice_hof(psboard: PsBoard) -> Callable:
    def _module_choice(val):
        if len(psboard.tab.children) < 3:
            psboard.tab.children += (psboard.current_module,)
        psboard.current_module.module_name = psboard.htable.value[
            len(psboard.htable.sensitive_css_class) + 1 :
        ]
        psboard.current_module.selection_changed = True
        psboard.tab.set_title(2, psboard.current_module.module_name)
        psboard.tab.selected_index = 2

    return _module_choice


# pylint: disable=too-many-ancestors,too-many-instance-attributes
class PsBoard(ipw.VBox):  # type: ignore
    def __init__(
        self,
        scheduler: Scheduler,
        order: Literal["asc", "desc"] = "asc",
        refresh_rate: int = 5,
    ):
        global debug_console  # pylint: disable=global-statement
        self._order: Literal["asc", "desc"] = order
        self.scheduler = scheduler
        self.refresh_rate = refresh_rate
        self.last_refresh = 0
        self._cache: Any = None
        self._cache_js: Any = None
        self.cpanel = ControlPanel(scheduler)
        self.current_module = ModuleWg(self, debug_console)
        self.mgraph = ModuleGraph()
        self.tab = ipw.Tab()
        self.tab.set_title(0, "Modules")
        self.tab.set_title(1, "Module graph")
        self.state: List[Any] = []
        self.last_update: List[Any] = []
        self.btns: List[WidgetType] = []
        self.msize: int = 0
        self.modules_changed: bool = True
        self.mgraph_changed: bool = True
        self.cols: List[str] = [
            "is_visualization",
            "id",
            "classname",
            "state",
            "last_update",
            "order",
        ]
        self.htable = SensitiveHTML(layout=ipw.Layout(height="500px", overflow="auto"))
        self.htable.observe(module_choice_hof(self), "value")
        # self.refresh_event = None
        self.other_coros: List[Coroutine[Any, Any, None]] = []
        self.vis_register: Dict[str, List[WidgetType]] = defaultdict(list)
        # commons.update(tab=self.tab, scheduler=self.scheduler)
        super().__init__([self.cpanel, self.tab, debug_console])
        self.scheduler.on_tick(self._refresh_proc)
        self.scheduler.on_change(self._change_proc)

    async def _refresh_proc(self, scheduler: Scheduler, run_number: int) -> None:
        assert scheduler is self.scheduler
        self.last_refresh += 1
        if self.last_refresh < self.refresh_rate:
            return
        self.last_refresh = 0
        json_ = self.scheduler.to_json(short=False)
        self._cache = JSONEncoderNp.dumps(json_, skipkeys=True)
        self._cache_js = None
        await self.refresh()

    async def _change_proc(
        self, scheduler: Scheduler, added: Set[Module], deleted: Set[Module]
    ) -> None:
        assert scheduler is self.scheduler
        # print("Dataflow changed")
        self.modules_changed = True
        self.mgraph_changed = True
        json_ = self.scheduler.to_json(short=False)
        self._cache = JSONEncoderNp.dumps(json_, skipkeys=True)
        self._cache_js = None
        await self.refresh()

    async def make_table_index(self, modules: List[Dict[str, JSon]]) -> None:
        modules = sorted(
            modules,
            key=lambda x: cast(int, x["order"]),
            reverse=(self._order == "desc"),
        )
        if not self.htable.html:
            await update_widget(self.htable, "sensitive_css_class", "ps-row-btn")
        if self.modules_changed:
            tmpl = Template(INDEX_TEMPLATE)
            html = tmpl.render(modules=modules, cols=self.cols)
            await update_widget(self.htable, "html", html)
            self.modules_changed = False
        else:
            data: Dict[str, Any] = {}
            for m in modules:
                for c in self.cols:
                    dataid = f"ps-cell_{m['id']}_{c}"
                    if c == "is_visualization":
                        # Show an Unicode eye next to visualizations
                        if cast(str, m["id"]) in self.vis_register:
                            content = "\U0001F441"
                        else:
                            content = " "
                        data[dataid] = content
                    else:
                        data[dataid] = m[c]
            await update_widget(self.htable, "data", data)

    def register_visualisation(
        self,
        widget: WidgetType,
        module: Module,
        label: str = "Visualisation",
        glue: Optional[
            Callable[[WidgetType, Module], Iterable[Coroutine[Any, Any, None]]]
        ] = None,
    ) -> None:
        """
        called from notebook

        if module_class is None and module_id is None:
            raise ValueError("One and only one of 'module_class' "
                              "and 'module_id' args must be defined")
        if not(module_class is None or module_id is None):
            raise ValueError("One and only one of 'module_class' "
            "and 'module_id' args must be defined")
        """
        linkable = hasattr(widget, "link_module")
        if not linkable and glue is None:
            raise ValueError(
                "Registering a visualisation requires a linkable "
                "widget (i.e. which implements the "
                "'link_module' interface) or 'glue' arg to be "
                "provides with a valid 'glue' function"
            )
        if glue is not None:
            self.other_coros += glue(widget, module)
        else:
            self.other_coros += widget.link_module(module, refresh=False)
        self.vis_register[module.name].append((widget, label))

    @property
    def coroutines(self) -> List[Coroutine[Any, Any, Any]]:
        return []

    async def refresh(self) -> None:
        if self._cache is None:
            return
        if self._cache_js is None:
            self._cache_js = JSONEncoderNp.loads(self._cache)
        json_ = self._cache_js
        # self.cpanel.run_nb.value = str(json_['run_number'])
        await update_widget(self.cpanel.run_nb, "value", str(json_["run_number"]))
        if self.tab.selected_index == 0:
            await self.make_table_index(json_["modules"])
        elif self.tab.selected_index == 1 and self.mgraph_changed:
            # self.mgraph.data = self._cache
            await update_widget(self.mgraph, "data", self._cache)
            self.mgraph_changed = False
        else:
            assert len(self.tab.children) > 2
            # FIXME fix when the displayed module is not deleted
            module_name = self.current_module.module_name
            module_json = None
            m = None
            for i, m in enumerate(json_["modules"]):
                if m["id"] == module_name:
                    module_json = m
                    break
            if module_json is not None:
                await self.current_module.refresh(module_json)
        if len(self.tab.children) < 3:
            self.tab.children = [self.htable, self.mgraph]
        else:
            third = self.tab.children[2]
            self.tab.children = [self.htable, self.mgraph, third]
