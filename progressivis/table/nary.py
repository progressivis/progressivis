"Base Module for modules supporting a variable number in input slots."
from __future__ import annotations

from progressivis.table.module import PTableModule
from progressivis.core.module import ReturnRunStep
from progressivis.table import BasePTable
from progressivis.core.slot import SlotDescriptor

from typing import List, Optional, Any


class NAry(PTableModule):
    "Base class for modules supporting a variable number of input slots."
    inputs = [SlotDescriptor("table", type=BasePTable, required=True, multiple=True)]

    def __init__(self, nary: str = "table", **kwds: Any) -> None:
        super(NAry, self).__init__(**kwds)
        self.nary = nary

    def predict_step_size(self, duration: float) -> int:
        return 1

    def get_input_slot_multiple(self, name: Optional[str] = None) -> List[str]:
        if name is None:
            name = self.nary
        return super(NAry, self).get_input_slot_multiple(name)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:  # pragma no cover
        raise NotImplementedError("run_step not defined")
