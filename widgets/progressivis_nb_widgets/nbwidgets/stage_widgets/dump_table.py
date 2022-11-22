from .utils import (
    stage_register,
    make_chaining_box,
    dongle_widget,
    set_child, ChainingVBox
)
from ..slot_wg import SlotWg

from progressivis.core.scheduler import Scheduler  # type: ignore

from typing import (
    Any as AnyType,
    Dict,
)


class DumpTableW(ChainingVBox):
    def __init__(self, ctx: Dict[str, AnyType]) -> None:
        super().__init__(ctx)
        self.dag_running()
        sl_wg = SlotWg(self._input_module, self._input_slot)
        self.children = (sl_wg, dongle_widget())
        set_child(self, 1, make_chaining_box(self))
        self._input_module.scheduler().on_tick(self._refresh_proc)

    async def _refresh_proc(self, scheduler: Scheduler, run_number: int) -> None:
        await self.children[0].refresh()

    def get_underlying_modules(self):
        return []


stage_register["Dump_table"] = DumpTableW
