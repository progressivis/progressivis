from __future__ import annotations

from progressivis.core.module import Module, ReturnRunStep, def_input, def_output
from ..utils.psdict import PDict
from .table import PTable

from typing import Any


@def_input("dict_", PDict, required=False)
@def_output("result", PTable)
class Dict2PTable(Module):
    """
    dict to table convertor

    Slots:
        dict_ : slot providing a PDict
    Args:
        kwds : argument to pass to the join function
    """

    def __init__(self, **kwds: Any) -> None:
        super().__init__(**kwds)

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        dict_slot = self.get_input_slot("dict_")
        assert dict_slot is not None
        dict_ = dict_slot.data()
        if dict_ is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if not (
            dict_slot.created.any()
            or dict_slot.updated.any()
            or dict_slot.deleted.any()
        ):
            return self._return_run_step(self.state_blocked, steps_run=0)
        dict_slot.created.next()
        dict_slot.updated.next()
        dict_slot.deleted.next()
        if self.result is None:
            self.result = PTable(name=None, dshape=dict_.dshape)
        if len(self.result) == 0:  # or history:
            self.result.append(dict_.as_row)
        else:
            self.result.loc[0] = dict_.array
        return self._return_run_step(self.next_state(dict_slot), steps_run=1)
