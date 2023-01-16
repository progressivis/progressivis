from __future__ import annotations

from progressivis.core.utils import Dialog
from progressivis.core.slot import SlotDescriptor
from progressivis.core.module import ReturnRunStep
from progressivis.table.module import PTableModule
from progressivis.utils.psdict import PDict

from typing import Any


class MergeDict(PTableModule):
    """
    Binary join module to join two dict and return a third one.

    Slots:
        first : PTable module producing the first dict to join
        second : PTable module producing the second dict to join
    Args:
        kwds : argument to pass to the join function
    """

    inputs = [
        SlotDescriptor("first", type=PDict, required=True),
        SlotDescriptor("second", type=PDict, required=True),
    ]

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)
        self._dialog = Dialog(self)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        first_slot = self.get_input_slot("first")
        # first_slot.update(run_number)
        second_slot = self.get_input_slot("second")
        assert first_slot is not None and second_slot is not None
        first_dict = first_slot.data()
        second_dict = second_slot.data()
        if first_dict is None or second_dict is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        # second_slot.update(run_number)
        first_slot.created.next()
        second_slot.created.next()
        first_slot.updated.next()
        second_slot.updated.next()
        first_slot.deleted.next()
        second_slot.deleted.next()
        if self.result is None:
            self.result = PDict(**first_dict, **second_dict)
        else:
            self.psdict.update(first_dict)
            self.psdict.update(second_dict)
        return self._return_run_step(self.next_state(first_slot), steps_run=1)
