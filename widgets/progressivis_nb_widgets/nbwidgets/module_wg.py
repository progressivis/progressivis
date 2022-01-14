from __future__ import annotations

import ipywidgets as ipw  # type: ignore

from .utils import update_widget
from .slot_wg import SlotWg
from .json_html import JsonHTML

from typing import Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .psboard import PsBoard


debug_console = None  # ipw.Output()
"""
{"table": [{"output_name": "table", "output_module": "csv_loader_1", "input_name": "df", "input_module": "every_1"}, {"output_name": "table", "output_module": "csv_loader_1", "input_name": "table", "input_module": "histogram_index_1"}, {"output_name": "table", "output_module": "csv_loader_1", "input_name": "table", "input_module": "histogram_index_2"}, {"output_name": "table", "output_module": "csv_loader_1", "input_name": "table", "input_module": "histogram_index_3"}, {"output_name": "table", "output_module": "csv_loader_1", "input_name": "table", "input_module": "histogram_index_4"}], "_trace": null}
## range_query_2d
{"min": [{"output_name": "min", "output_module": "range_query2d_2", "input_name": "table.00.03", "input_module": "mc_histogram2_d_1"}, {"output_name": "min", "output_module": "range_query2d_2", "input_name": "table.00.03", "input_module": "mc_histogram2_d_2"}], "max": [{"output_name": "max", "output_module": "range_query2d_2", "input_name": "table.00.04", "input_module": "mc_histogram2_d_1"}, {"output_name": "max", "output_module": "range_query2d_2", "input_name": "table.00.04", "input_module": "mc_histogram2_d_2"}], "table": [{"output_name": "table", "output_module": "range_query2d_2", "input_name": "data", "input_module": "mc_histogram2_d_2"}, {"output_name": "table", "output_module": "range_query2d_2", "input_name": "table", "input_module": "sample_2"}], "_trace": null}

"""


class ModuleWg(ipw.Tab):  # type: ignore # pylint: disable=too-many-ancestors
    def __init__(self, board: PsBoard, dconsole: Optional[Any] = None) -> None:
        global debug_console  # pylint: disable=global-statement
        debug_console = dconsole
        self.children: Tuple[Any, ...]
        self._index = board
        self._main = JsonHTML()
        self.module_name: Optional[str] = None
        self.selection_changed = False
        self._output_slots = ipw.Tab()
        super().__init__([self._main, self._output_slots])
        self.set_title(0, "Main")
        self.set_title(1, "Output slots")

    async def refresh(self) -> None:
        _idx = self._index
        # pylint: disable=protected-access
        json_ = _idx._cache_js
        assert json_ is not None
        module_json = None
        m = None
        for i, m in enumerate(json_["modules"]):
            if m["id"] == self.module_name:
                module_json = m
                break
        assert module_json is not None
        self.set_title(0, self.module_name)
        await update_widget(self._main, "data", m)
        await update_widget(
            self._main,
            "config",
            dict(
                order=[
                    "classname",
                    "speed",
                    "debug",
                    "state",
                    "last_update",
                    "default_step_size",
                    "start_time",
                    "end_time",
                    "parameters",
                    "input_slots",
                ],
                sparkline=["speed"],
            ),
        )
        _selected_index = 0
        scheduler = self._index.scheduler
        assert scheduler is not None and self.module_name is not None
        if self.module_name not in scheduler:
            return
        module = scheduler[self.module_name]
        if self.selection_changed or not self._output_slots.children:
            # first refresh
            self.selection_changed = False
            self.selected_index = 0
            slots = [SlotWg(module, sl) for sl in m["output_slots"].keys()] + [
                SlotWg(module, "_params")
            ]
            await update_widget(self._output_slots, "children", slots)
            if module.name in self._index.vis_register:
                for wg, label in self._index.vis_register[module.name]:
                    self.children = (wg,) + self.children
                    self.set_title(0, label)
                    self.set_title(1, "Main")
                    self.set_title(2, "Output slots")
            elif len(self.children) > 2:
                self.children = self.children[1:]
                self.set_title(0, module.name)
                self.set_title(1, "Output slots")
        else:
            _selected_index = self._output_slots.selected_index
        for i, k in enumerate(m["output_slots"].keys()):
            self._output_slots.set_title(i, k)
            await self._output_slots.children[i].refresh()
        i += 1
        self._output_slots.set_title(i, "_params")
        await self._output_slots.children[i].refresh()
        self._output_slots.selected_index = _selected_index
