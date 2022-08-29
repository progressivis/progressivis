import ipywidgets as ipw
from .utils import (
    stage_register,
    make_chaining_box,
    dongle_widget,
    set_child, ChainingWidget
)
from ..slot_wg import SlotWg

from progressivis.table.module import TableModule
from progressivis.core.scheduler import Scheduler

from typing import (
    Any as AnyType,
    Dict,
)


class DumpTableW(ipw.VBox, ChainingWidget):
    def __init__(
        self,
        frame: AnyType,
        dtypes: Dict[str, AnyType],
        input_module: TableModule,
        input_slot: str = "result",
    ) -> None:
        super().__init__(frame=frame,
                         dtypes=dtypes,
                         input_module=input_module,
                         input_slot=input_slot)
        sl_wg = SlotWg(input_module, input_slot)
        self.children = (sl_wg, dongle_widget())
        set_child(self, 1, make_chaining_box(self))
        self._input_module.scheduler().on_tick(self._refresh_proc)

    async def _refresh_proc(self, scheduler: Scheduler, run_number: int) -> None:
        await self.children[0].refresh()

    def get_underlying_modules(self):
        return []


stage_register["Dump table"] = DumpTableW
