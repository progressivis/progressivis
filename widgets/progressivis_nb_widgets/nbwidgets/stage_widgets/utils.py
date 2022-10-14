import ipywidgets as ipw  # type: ignore
from progressivis.table.dshape import dataframe_dshape  # type: ignore
from progressivis.vis import DataShape  # type: ignore
from progressivis.core import Sink  # type: ignore
from progressivis.core.utils import normalize_columns  # type: ignore
from IPython.display import Javascript, display  # type: ignore
from collections import defaultdict
import dagWidget  # type: ignore
from typing import (
    Any as AnyType,
    Optional,
    Dict,
    Set,
    List,
    Callable,
)


dag_widget = None


def get_dag():
    global dag_widget
    if dag_widget is None:
        dag_widget = dagWidget.HelloWorld()
    return dag_widget


WidgetType = AnyType


def get_param(d, key, default):
    if key not in d:
        return default
    val = d[key]
    if not val:
        return default
    return val


def set_child(wg, i, child, title=""):
    children = list(wg.children)
    children[i] = child
    wg.children = tuple(children)
    if title:
        wg.set_title(i, title)


def append_child(wg, child, title=""):
    children = list(wg.children)
    last = len(children)
    children.append(child)
    wg.children = tuple(children)
    if title:
        wg.set_title(last, title)


def dongle_widget(v="dongle"):
    wg = ipw.Label(v)
    wg.layout.visibility = "hidden"
    return wg


class HandyTab(ipw.Tab):
    def set_next_title(self, name: str) -> None:
        pos = len(self.children) - 1
        self.set_title(pos, name)

    def get_titles(self) -> List[str]:
        return [self.get_title(pos) for pos in range(len(self.children))]

    def set_tab(self, title: str, wg: WidgetType, overwrite: bool = True) -> None:
        all_titles = self.get_titles()
        if title in all_titles:
            if not overwrite:
                return
            pos = all_titles.index(title)
            children_ = list(self.children)
            children_[pos] = wg
            self.children = tuple(children_)
        else:
            self.children += (wg,)
            self.set_next_title(title)

    def remove_tab(self, title):
        all_titles = self.get_titles()
        if title not in all_titles:
            return
        pos = all_titles.index(title)
        children_ = list(self.children)
        children_ = children_[:pos] + children_[pos + 1:]
        titles_ = all_titles[:pos] + all_titles[pos + 1:]
        self.children = tuple(children_)
        for i, t in enumerate(titles_):
            self.set_title(i, t)

    def get_selected_title(self):
        if self.selected_index is None:
            return None
        return self.get_title(self.selected_index)

    def get_selected_child(self):
        if self.selected_index is None:
            return None
        return self.children[self.selected_index]


class TreeTab(HandyTab):
    def __init__(
        self, upper: Optional["TreeTab"], known_as: str, *args: AnyType, **kw: AnyType
    ) -> None:
        super().__init__(*args, **kw)
        self.upper = upper
        self.known_as = known_as
        self.mod_dict: Dict[str, Set[str]] = {}

    def is_visible(self, sel):
        if self.get_selected_title() != sel:
            return False
        if self.upper is None:
            return True
        return self.upper.is_visible(self.known_as)


def get_schema(sniffer) -> AnyType:
    params = sniffer.params
    usecols = params.get("usecols")
    parse_dates = get_param(params, "parse_dates", [])

    def _ds(col, dt):
        if col in parse_dates:
            return "datetime64"
        return dataframe_dshape(dt)
    norm_cols = dict(zip(sniffer._df.columns, normalize_columns(sniffer._df.columns)))
    dtypes = {col: _ds(col, dt) for (col, dt) in sniffer._df.dtypes.to_dict().items()}
    if usecols is not None:
        dtypes = {norm_cols[col]: dtypes[col] for col in usecols}
    else:
        dtypes = {norm_cols[col]: t for (col, t) in dtypes.items()}
    return dtypes


def make_button(
    label: str, disabled: bool = False, cb: Optional[Callable] = None
) -> ipw.Button:
    btn = ipw.Button(
        description=label,
        disabled=disabled,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip=label,
        icon="check",  # (FontAwesome names without the `fa-` prefix)
    )
    if cb is not None:
        btn.on_click(cb)
    return btn


def make_guess_types_toc2(obj, sel, fun):
    def _guess(m, run_number):
        global parent_dtypes
        if m.result is None:
            return
        parent_dtypes = {k: "datetime64" if str(v)[0] == "6"
                         else v for (k, v) in m.result.items()}
        obj._output_dtypes = parent_dtypes
        fun(obj, sel.value)
        with m.scheduler() as dataflow:
            deps = dataflow.collateral_damage(m.name)
            dataflow.delete_modules(*deps)

    return _guess


stage_register: Dict[str, AnyType] = {}
parent_widget = None
parent_dtypes = None
# last_created = None
widget_by_id = {}
widget_by_key = {}
widget_numbers: Dict[str, int] = defaultdict(int)


class _Dag:
    def __init__(self, label, number, dag, alias=""):
        self._label = label
        if alias:
            self._number = 0
        else:
            self._number = number
        self._dag = dag
        self._alias = alias


def create_stage_widget(key):
    # global last_created
    obj = parent_widget
    dtypes = obj._output_dtypes
    if dtypes is None:
        dtypes = parent_dtypes
    dag = _Dag(label=key, number=widget_numbers[key], dag=get_dag())
    stage = stage_register[key](obj, dtypes, obj._output_module, dag=dag)
    widget_numbers[key] += 1
    assert obj not in obj.subwidgets
    obj.subwidgets.append(stage)
    widget_by_key[(key, stage.number)] = stage
    widget_by_id[id(stage)] = stage
    return stage


def create_loader_widget(key, ftype, alias):
    obj = parent_widget
    dtypes = None
    assert obj not in obj.subwidgets
    dag = _Dag(label=key, number=widget_numbers[key], dag=get_dag(), alias=alias)
    if ftype == "csv":
        from .csv_loader import CsvLoaderW

        stage = CsvLoaderW(obj, dtypes, obj._output_module, dag=dag)
    else:
        assert ftype == "parquet"
        from .parquet_loader import ParquetLoaderW

        stage = ParquetLoaderW(obj, dtypes, obj._output_module, dag=dag)
    widget_numbers[key] += 1
    obj.subwidgets.append(stage)
    widget_by_id[id(stage)] = stage
    if alias:
        widget_by_key[(alias, 0)] = stage
    else:
        widget_by_key[(key, stage.number)] = stage
    return stage


def get_widget_by_id(key):
    return widget_by_id[key]


def get_widget_by_key(key, num):
    return widget_by_key[(key, num)]


def _make_btn_start_toc2(obj: AnyType, sel: AnyType, fun) -> Callable:
    def _cbk(btn: ipw.Button) -> None:
        global parent_widget
        parent_widget = obj
        assert parent_widget
        if obj._output_dtypes is None:
            s = obj._output_module.scheduler()
            with s:
                ds = DataShape(scheduler=s)
                ds.input.table = obj._output_module.output.result
                ds.on_after_run(make_guess_types_toc2(obj, sel, fun))
                sink = Sink(scheduler=s)
                sink.input.inp = ds.output.result
        else:
            fun(obj, sel.value)

    return _cbk


def _make_btn_start_loader(obj: AnyType, ftype: str, alias: WidgetType) -> Callable:
    def _cbk(btn: ipw.Button) -> None:
        global parent_widget
        parent_widget = obj
        assert parent_widget
        add_new_loader(obj, ftype, alias.value)
        alias.value = ""
    return _cbk


remove_js_func = """
(function(){{
  let indices = [];
  IPython.notebook.get_cells().forEach( function(cell) {{
    if (cell.metadata !== undefined){{
      if(cell.metadata.progressivis_tag === "{tag}"){{
        cell.metadata.editable = true;
        cell.metadata.deletable = true;
        let i = IPython.notebook.find_cell_index(cell);
        indices.push(i);
      }}
    }}
  }});
  let uIndices = [...new Set(indices)];
  IPython.notebook.delete_cells(uIndices);
}})();
"""


def remove_tagged_cells(tag):
    s = remove_js_func.format(tag=tag)
    display(Javascript(s))


def _remove_subtree(obj):
    for sw in obj.subwidgets:
        _remove_subtree(sw)
    tag = id(obj)
    remove_tagged_cells(tag)
    if obj.parent is not None:
        obj.parent.subwidgets.remove(obj)
    if tag in widget_by_id:
        del widget_by_id[tag]
        del widget_by_key[(obj.label, obj.number)]
    obj.delete_underlying_modules()


def make_remove(obj):
    def _cbk(btn: ipw.Button) -> None:
        _remove_subtree(obj)

    return _cbk


def make_chaining_box(obj):
    fnc = _make_btn_start_toc2
    sel = ipw.Dropdown(
        options=[""] + list(stage_register.keys()),
        value="",
        # rows=10,
        description="Next stage",
        disabled=False,
    )
    btn = make_button("Chain it", disabled=True, cb=fnc(obj, sel, add_new_stage))
    del_btn = make_button("Remove subtree", disabled=False, cb=make_remove(obj))

    def _on_sel_change(change):
        if change["new"]:
            btn.disabled = False
        else:
            btn.disabled = True

    sel.observe(_on_sel_change, names="value")
    return ipw.HBox([sel, btn, del_btn])


def make_loader_box(obj, ftype="csv"):
    fnc = _make_btn_start_loader
    alias_inp = ipw.Text(value='',
                         placeholder='optional alias',
                         description=f"{ftype.upper()} loader:",
                         disabled=False)
    btn = make_button(
        "Create", disabled=False, cb=fnc(obj, ftype, alias_inp)
    )
    return ipw.HBox([alias_inp, btn])


cleanup_js_func = """
(function(){{
  let indices = [];
  IPython.notebook.get_cells().forEach( function(cell) {{
    if (cell.metadata !== undefined){{
      if(cell.metadata.progressivis_tag !== undefined){{
        cell.metadata.editable = true;
        cell.metadata.deletable = true;
        let i = IPython.notebook.find_cell_index(cell);
        indices.push(i);
      }}
    }}
  }});
  let uIndices = [...new Set(indices)];
  IPython.notebook.delete_cells(uIndices);
}})();
"""


def cleanup_cells():
    display(Javascript(cleanup_js_func))


js_func_toc = """
(function(){{
  let i = -1;
  IPython.notebook.get_cells().forEach( function(cell) {{
    if (cell.metadata !== undefined){{
      if(cell.metadata.progressivis_tag === "{prev}"){{
        cell.metadata.editable = true;
        cell.metadata.deletable = true;
        i = IPython.notebook.find_cell_index(cell);
      }}
    }}
  }});
  if(i<0){{
   i = IPython.notebook.get_cell_elements().length;
  }} else {{
    i = i+1;
  }}
  IPython.notebook.insert_cell_at_index("markdown", i).set_text("{md}");
  IPython.notebook.select(i);
  IPython.notebook.execute_cell(i);
  let meta = {{
    "trusted": true,
    "editable": false,
    "deletable": false,
    "progressivis_tag": "{tag}"
   }};
  IPython.notebook.get_cell(i).metadata = meta;
  IPython.notebook.insert_cell_at_index("code", i+1).set_text("{code}");
  IPython.notebook.select(i+1);
  IPython.notebook.execute_cell(i+1);
  IPython.notebook.get_cell(i+1).metadata = meta;
}})();
"""

js_func_cell_index = """
(function(){{
  IPython.notebook.insert_cell_at_index("{kind}", {index}).set_text("{text}");
  IPython.notebook.select({index});
  IPython.notebook.execute_cell({index});
  let meta = {{
    "trusted": true,
    "editable": false,
    "deletable": false,
    "progressivis_tag": "{tag}"
   }};
  IPython.notebook.get_cell({index}).metadata = meta;
  }})();
"""


def insert_cell_at_index(kind, text, index, tag):
    display(
        Javascript(
            js_func_cell_index.format(kind=kind, text=text, index=index, tag=tag)
        )
    )


def get_previous(obj):
    if not obj.subwidgets:
        return obj
    return get_previous(obj.subwidgets[-1])


new_stage_cell_0 = "Constructor.widget('{key}')"
new_stage_cell = "Constructor.widget('{key}', {num})"


def add_new_stage(parent, title):
    previous = get_previous(parent)
    prev = id(previous)
    stage = create_stage_widget(title)
    tag = id(stage)
    n = stage.number
    md = "## " + title + (f"[{n}]" if n else "")
    code = new_stage_cell.format(key=title, num=n)
    s = js_func_toc.format(prev=prev, tag=tag, md=md, code=code)
    display(Javascript(s))


def add_new_loader(parent, ftype, alias):
    title = f"{ftype.upper()} loader"
    prev = "no_previous"
    stage = create_loader_widget(title, ftype, alias)
    tag = id(stage)
    n = stage.number
    if alias:
        md = f"## {alias}"
        code = new_stage_cell_0.format(key=alias)
    else:
        md = "## " + title + (f"[{n}]" if n else "")
        if n:
            code = new_stage_cell.format(key=title, num=n)
        else:
            code = new_stage_cell_0.format(key=title)
    s = js_func_toc.format(prev=prev, tag=tag, md=md, code=code)
    display(Javascript(s))


class ChainingWidget:
    def __init__(self, *args, **kw):
        assert "parent" in kw
        self.parent = kw["parent"]
        assert "dtypes" in kw
        self._dtypes = kw["dtypes"]
        assert "input_module" in kw
        self._input_module = kw["input_module"]
        self._input_slot = kw.get("input_slot", "result")
        self._output_module = self._input_module
        self._output_slot = self._input_slot
        if self._dtypes is not None:  # i.e. not a loader
            self._output_dtypes = None
        self._dag = kw["dag"]
        self.subwidgets = []
        self.managed_modules = []

    def get_underlying_modules(self):
        raise NotImplementedError()

    def delete_underlying_modules(self):
        managed_modules = self.get_underlying_modules()
        if not managed_modules:
            return
        with self._input_module.scheduler() as dataflow:
            # for m in obj.managed_modules:
            deps = dataflow.collateral_damage(*managed_modules)
            dataflow.delete_modules(*deps)

    def dag_register(self):
        return self.dag.registerWidget(self, self.title,
                                       self.title,
                                       self.dom_id,
                                       [self.parent.title])

    def dag_running(self, progress=0):
        self.dag.updateSummary(self.title, {"progress": progress, "status": "RUNNING"})

    @property
    def dag(self):
        return self._dag._dag

    @property
    def dom_id(self):
        return self.title.replace(" ", "-")

    @property
    def label(self):
        return self._dag._label

    @property
    def number(self):
        return self._dag._number

    @property
    def _frame(self):
        return self.parent._frame+1

    @property
    def title(self):
        if self._dag._alias:
            return self._dag._alias
        return f"{self.label}[{self.number}]" if self.number else self.label
