import ipywidgets as ipw

from .templates import *
from .utils import *
from .sensitive_html import SensitiveHTML
import weakref
import json
debug_console = None #ipw.Output()
"""
{"table": [{"output_name": "table", "output_module": "csv_loader_1", "input_name": "df", "input_module": "every_1"}, {"output_name": "table", "output_module": "csv_loader_1", "input_name": "table", "input_module": "histogram_index_1"}, {"output_name": "table", "output_module": "csv_loader_1", "input_name": "table", "input_module": "histogram_index_2"}, {"output_name": "table", "output_module": "csv_loader_1", "input_name": "table", "input_module": "histogram_index_3"}, {"output_name": "table", "output_module": "csv_loader_1", "input_name": "table", "input_module": "histogram_index_4"}], "_trace": null}
## range_query_2d
{"min": [{"output_name": "min", "output_module": "range_query2d_2", "input_name": "table.00.03", "input_module": "mc_histogram2_d_1"}, {"output_name": "min", "output_module": "range_query2d_2", "input_name": "table.00.03", "input_module": "mc_histogram2_d_2"}], "max": [{"output_name": "max", "output_module": "range_query2d_2", "input_name": "table.00.04", "input_module": "mc_histogram2_d_1"}, {"output_name": "max", "output_module": "range_query2d_2", "input_name": "table.00.04", "input_module": "mc_histogram2_d_2"}], "table": [{"output_name": "table", "output_module": "range_query2d_2", "input_name": "data", "input_module": "mc_histogram2_d_2"}, {"output_name": "table", "output_module": "range_query2d_2", "input_name": "table", "input_module": "sample_2"}], "_trace": null}

"""
class ModuleWg(ipw.Tab):
    def __init__(self, board, dconsole=None):
        global debug_console
        debug_console = dconsole
        self._index = board #weakref.ref(board)
        self._main = ipw.HTML()
        self.module_name = None
        self._output_slots = ipw.Tab()
        super().__init__([self._main, self._output_slots])
        self.set_title(0, 'Main')
        self.set_title(1, 'Output slots')

    def refresh(self):
        #foo = 7/0
        _idx = self._index
        json_ = _idx._cache_js
        assert json_ is not None
        module_json = None
        for i, m in enumerate(json_['modules']):
                if m['id']==self.module_name:
                     module_json = m
                     break
        assert module_json is not None
        self.set_title(0, self.module_name)
        self._main.value = layout_dict(m, order=["classname",
                                   "speed",
                                   #"output_slots",
                                   "debug",
                                   "state",
                                   "last_update",
                                   "default_step_size",
                                   "start_time",
                                   "end_time",
                                   "parameters",
                                   "input_slots"])
        _selected_index = None
        if not self._output_slots.children: # first refresh
            self._output_slots.children = [ipw.HTML() for _ in range(len(m["output_slots"]))]
        else:
            _selected_index = self._output_slots.selected_index
        for i, (k, v) in enumerate(m["output_slots"].items()):
            self._output_slots.set_title(i, k)
            v = "None"
            module = self._index.scheduler.modules()[self.module_name]
            if module is None:
                self._output_slots.children[i].value = v
            else:
                df = module.get_data(k)
                self._output_slots.children[i].value = str(df)
        self._output_slots.selected_index = _selected_index

