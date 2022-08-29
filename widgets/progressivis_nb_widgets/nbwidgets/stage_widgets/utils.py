# flake8: noqa


import ipywidgets as ipw
from progressivis.table.dshape import dataframe_dshape
from progressivis.table.module import TableModule
from progressivis.vis import DataShape
from progressivis.core import Sink
from abc import ABCMeta, abstractmethod
from IPython.display import Javascript

from typing import (
    Any as AnyType,
    cast,
    Optional,
    Dict,
    Set,
    Union,
    List,
    Tuple,
    Callable,
)

WidgetType = AnyType

def get_param(d, key, default):
    if key not in d:
        return default
    val = d[key]
    if not val:
        return default
    return val

def set_child(wg, i, child):
    children = list(wg.children)
    children[i] = child
    wg.children = tuple(children)

def append_child(wg, child, title=""):
    children = list(wg.children)
    last = len(children)
    children.append(child)
    wg.children = tuple(children)
    if title:
        wg.set_title(last, title)

def dongle_widget(v="dongle"):
    wg = ipw.Label(v)
    wg.layout.visibility = 'hidden'
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
        children_ = children_[:pos] + children_[pos + 1 :]
        titles_ = all_titles[:pos] + all_titles[pos + 1 :]
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
        return  dataframe_dshape(dt)
    dtypes = {col: _ds(col, dt) for (col, dt) in
              sniffer._df.dtypes.to_dict().items()}
    if usecols is not None:
        dtypes = {col: dtypes[col] for col in usecols}
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

def make_guess_types(obj, sel):
    def _guess(m, run_number):
        if m.result is None:
            return
        stage = stage_register[sel.value](obj._frame, m.result, obj._output_module)
        append_child(obj._frame, stage, sel.value)

    return _guess

#new_stage_cell = "display(AnalysisToC.create_stage_widget('{key}'))"

new_stage_cell = "display(AnalysisToC.widget_by_id({key}))"


def make_guess_types_toc2(obj, sel):
    def _guess(m, run_number):
        global parent_dtypes
        if m.result is None:
            return
        parent_dtypes = m.result
        add_new_stage(obj,
                      sel.value)
    return _guess

stage_register = {}
parent_widget = None
parent_dtypes = None
#last_created = None
widget_by_id = {}

def create_stage_widget(key):
    #global last_created
    obj = parent_widget
    dtypes = obj._output_dtypes
    if dtypes is None:
        dtypes = parent_dtypes
    stage = stage_register[key](obj._frame+1, dtypes, obj._output_module)
    assert obj not in obj.subwidgets
    obj.subwidgets.append(stage)
    stage.parent = obj
    #last_created = stage
    widget_by_id[id(stage)] = stage
    return stage

def get_widget_by_id(key):
    return widget_by_id[key]

#def current_widget():
#    return last_created

def _make_btn_start(obj: AnyType, sel: AnyType) -> Callable:
        def _cbk(btn: ipw.Button) -> None:
            if obj._output_dtypes is None:
                s = obj._output_module.scheduler()
                with s:
                    ds = DataShape(scheduler=s)
                    ds.input.table = obj._output_module.output.result
                    ds.on_after_run(make_guess_types(obj, sel))
                    sink = Sink(scheduler=s)
                    sink.input.inp = ds.output.result
            else:
                stage = stage_register[sel.value](obj._frame, obj._output_dtypes, obj._output_module)
                append_child(obj._frame, stage, sel.value)
        return _cbk

def _make_btn_start_toc2(obj: AnyType, sel: AnyType) -> Callable:
        def _cbk(btn: ipw.Button) -> None:
            global parent_widget
            parent_widget = obj
            if obj._output_dtypes is None:
                s = obj._output_module.scheduler()
                with s:
                    ds = DataShape(scheduler=s)
                    ds.input.table = obj._output_module.output.result
                    ds.on_after_run(make_guess_types_toc2(obj, sel))
                    sink = Sink(scheduler=s)
                    sink.input.inp = ds.output.result
            else:
                add_new_stage(obj,
                              sel.value)
        return _cbk


remove_js_func = """
var p10s_rem = function(){{
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
    let uIndices = [...new Set(indices)];
    IPython.notebook.delete_cells(uIndices);
  }});
}};
p10s_rem();
"""

def remove_tagged_cells(tag):
    s = remove_js_func.format(tag=tag)
    display(Javascript(s))

def _remove_subtree(obj):
    for sw in obj.subwidgets:
        _remove_subtree(sw)
    tag = id(obj)
    print("remove tagged cells", obj, tag)
    remove_tagged_cells(tag)
    if obj.parent is not None:
        obj.parent.subwidgets.remove(obj)
    if tag in widget_by_id:
        del  widget_by_id[tag]
    obj.delete_underlying_modules()



def make_remove(obj):
    def _cbk(btn: ipw.Button) -> None:
        _remove_subtree(obj)
    return _cbk

def make_chaining_box(obj):
    fnc = _make_btn_start_toc2 if isinstance(obj._frame, int) else _make_btn_start
    sel = ipw.Dropdown(
        options=[""]+list(stage_register.keys()),
        value="",
        # rows=10,
        description="Next stage",
        disabled=False
    )
    btn = make_button("Chain it", disabled=True, cb=fnc(obj, sel))
    del_btn = make_button("Remove subtree", disabled=False, cb=make_remove(obj))
    def _on_sel_change(change):
        if change['new']:
            btn.disabled = False
        else:
            btn.disabled = True
    sel.observe(_on_sel_change, names='value')
    return ipw.HBox([sel, btn, del_btn])

cleanup_js_func = """
var p10s_cleanup = function(){{
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
    let uIndices = [...new Set(indices)];
    IPython.notebook.delete_cells(uIndices);
  }});
}};
p10s_cleanup();
"""

def cleanup_cells():
    display(Javascript(cleanup_js_func))



js_func = """
var p10s = function(){{
  let i = IPython.notebook.get_cell_elements().length;
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
}};
p10s();
"""

js_tag_func = """
var p10s_tag = function(){{
  let indices = [];
  IPython.notebook.get_cells().forEach( function(cell) {{
    if (cell.metadata !== undefined){{
      if(cell.metadata.progressivis_tag === "last_created"){{
        cell.metadata.editable = false;
        cell.metadata.deletable = false;
        cell.metadata.progressivis_tag = "{tag}";
      }}
    }}
  }});
}};
p10s_tag();
"""




js_func_cell_index = """
var p10s_cell_index = function(){{
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
  }}
p10s_cell_index();
"""

def insert_cell_at_index(kind, text, index, tag):
     display(Javascript(js_func_cell_index.format(kind=kind, text=text, index=index, tag=tag)))

def add_new_stage(parent, title):
    level = parent._frame + 1
    stage = create_stage_widget(title)
    tag = id(stage)
    md = "#"*level + " " + title
    code = new_stage_cell.format(key=tag)
    display(Javascript(js_func.format(tag=tag, md=md, code=code)))
    #tag = id(last_created)
    #display(Javascript(js_tag_func.format(tag=tag)))

class ChainingWidget:
    def __init__(self, *args, **kw):
        assert "frame" in kw
        self._frame = kw["frame"]
        assert "dtypes" in kw
        self._dtypes = kw["dtypes"]
        assert "input_module" in kw
        self._input_module = kw["input_module"]
        self._input_slot = kw.get("input_slot", "result")
        self._output_module = self._input_module
        self._output_slot = self._input_slot
        if self._dtypes is not None:
            self._output_dtypes = self._dtypes
        self.parent = None
        self.subwidgets = []
        self.managed_modules = []

    def get_underlying_modules(self):
        raise NotImplementedError()

    def delete_underlying_modules(self):
        managed_modules = self.get_underlying_modules()
        print("underlying modules", managed_modules)
        if not managed_modules:
            return
        with self._input_module.scheduler() as dataflow:
            #for m in obj.managed_modules:
            deps = dataflow.collateral_damage(*managed_modules)
            dataflow.delete_modules(*deps)
