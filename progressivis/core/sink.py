"Sink Module, to keep table modules alive with no other output slots connected"
from __future__ import annotations

from progressivis.core.slot import SlotDescriptor
from progressivis.core.module import Module, ReturnRunStep

from typing import List, Any, Optional


class Sink(Module):
    "Base class for modules supporting a variable number of input slots."
    inputs = [SlotDescriptor("inp", type=None, required=True, multiple=True)]

    def __init__(self, slot_name: str = "inp", **kwds: Any) -> None:
        super(Sink, self).__init__(**kwds)
        self.slot_name = slot_name

    def predict_step_size(self, duration: float) -> int:
        return 1

    def get_input_slot_multiple(self, name: Optional[str] = None) -> List[str]:
        if name is None:
            name = self.slot_name
        return super(Sink, self).get_input_slot_multiple(name)

    def prepare_run(self, run_number: int) -> None:
        "Switch from zombie to terminated, or update slots."
        if self.state == Module.state_zombie:
            self.state = Module.state_terminated

    def is_ready(self) -> bool:
        self.state = Module.state_terminated
        return False

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:  # pragma no cover
        raise NotImplementedError("run_step not defined")
