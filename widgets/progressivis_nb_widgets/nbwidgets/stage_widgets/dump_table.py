from .utils import (
    stage_register,
    VBox
)
from ..slot_wg import SlotWg

from progressivis.core.scheduler import Scheduler


class DumpPTableW(VBox):
    def __init__(self) -> None:
        super().__init__()

    def init(self) -> None:
        self.dag_running()
        sl_wg = SlotWg(self.input_module, self.input_slot)
        self.children = (sl_wg,)
        self.input_module.scheduler().on_tick(self._refresh_proc)

    async def _refresh_proc(self, scheduler: Scheduler, run_number: int) -> None:
        await self.children[0].refresh()

    def get_underlying_modules(self):
        return []


stage_register["Dump_table"] = DumpPTableW
