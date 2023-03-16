from __future__ import annotations

import ipywidgets as ipw

from .utils import update_widget
from .slot_wg import SlotWg
from .json_html import JsonHTML

from typing import Any, Optional, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .psboard import PsBoard


debug_console = None  # ipw.Output()


class ModuleWg(ipw.Tab):  # pylint: disable=too-many-ancestors
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

    async def refresh(self, module_json: Dict[str, Any]) -> None:
        # _idx = self._index
        # # pylint: disable=protected-access
        # json_ = _idx._cache_js
        # assert json_ is not None
        assert self.module_name is not None
        self.set_title(0, self.module_name)
        await update_widget(self._main, "data", module_json)
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
            slots = [
                SlotWg(module, sl) for sl in module_json["output_slots"].keys()
            ] + [SlotWg(module, "_params")]
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
        for i, k in enumerate(module_json["output_slots"].keys()):
            self._output_slots.set_title(i, k)
            item = self._output_slots.children[i]
            assert hasattr(item, "refresh")
            await item.refresh()
        i += 1
        self._output_slots.set_title(i, "_params")
        item = self._output_slots.children[i]
        assert hasattr(item, "refresh")
        await item.refresh()
        self._output_slots.selected_index = _selected_index
