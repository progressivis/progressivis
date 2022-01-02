from __future__ import annotations

from progressivis.core.utils import Dialog
from progressivis.core.slot import SlotDescriptor

from progressivis.table.module import TableModule, ReturnRunStep
from progressivis.utils.psdict import PsDict


class MergeDict(TableModule):
    """
    Binary join module to join two dict and return a third one.

    Slots:
        first : Table module producing the first dict to join
        second : Table module producing the second dict to join
    Args:
        kwds : argument to pass to the join function
    """

    inputs = [
        SlotDescriptor("first", type=PsDict, required=True),
        SlotDescriptor("second", type=PsDict, required=True),
    ]

    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._dialog = Dialog(self)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        first_slot = self.get_input_slot("first")
        # first_slot.update(run_number)
        second_slot = self.get_input_slot("second")
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
            self.result = PsDict(**first_dict, **second_dict)
        else:
            self.psdict.update(first_dict)
            self.psdict.update(second_dict)
        return self._return_run_step(self.next_state(first_slot), steps_run=1)
