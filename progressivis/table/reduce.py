from __future__ import annotations

from progressivis import Scheduler, Slot
from progressivis.table.nary import NAry
from progressivis.table.module import TableModule

from typing import Type, Any, List


class Reduce(NAry):
    "Reduce binary modules over multiple inputs"

    @staticmethod
    def expand(
        binary_module: Type[TableModule],
        left_in: str,
        right_in: str,
        outp: str,
        slots: List[Slot],
        **binary_module_kwds: Any
    ) -> TableModule:
        if len(slots) < 2:
            raise ValueError("Reduce needs at least two unputs")
        scheduler = binary_module_kwds.get("scheduler")
        if scheduler is None:
            scheduler = Scheduler.default
            binary_module_kwds["scheduler"] = scheduler
        prev_slot = slots[0]
        with scheduler:
            for slot in slots[1:]:
                bin_mod = binary_module(**binary_module_kwds)
                bin_mod.input[left_in] = prev_slot
                bin_mod.input[right_in] = slot
                prev_slot = bin_mod.output[outp]
        return bin_mod
