from __future__ import annotations

from progressivis.core.utils import Dialog
from progressivis.core.module import Module, ReturnRunStep, def_input, def_output
from progressivis.utils.psdict import PDict

from typing import Any


@def_input("first", PDict)
@def_input("second", PDict)
@def_output("result", PDict)
class MergeDict(Module):
    """
    Binary join module to join two dict and return a third one.

    Slots:
        first : PDict module producing the first dict to join
        second : PDict module producing the second dict to join
    Args:
        kwds : argument to pass to the join function
    """

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
            self.result.update(first_dict)
            self.result.update(second_dict)
        return self._return_run_step(self.next_state(first_slot), steps_run=1)
