from __future__ import annotations

from progressivis.core.slot import SlotDescriptor
from .module import TableModule, ReturnRunStep
from ..utils.psdict import PsDict
from .table import Table


class Dict2Table(TableModule):
    """
    dict to table convertor

    Slots:
        dict_ : Table module producing the first table to join
    Args:
        kwds : argument to pass to the join function
    """

    inputs = [SlotDescriptor("dict_", type=PsDict, required=True)]

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        dict_slot = self.get_input_slot("dict_")
        # dict_slot.update(run_number)
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
            self.result = Table(name=None, dshape=dict_.dshape)
        if len(self.result) == 0:  # or history:
            self.table.append(dict_.as_row)
        else:
            self.table.loc[0] = dict_.array
        return self._return_run_step(self.next_state(dict_slot), steps_run=1)
