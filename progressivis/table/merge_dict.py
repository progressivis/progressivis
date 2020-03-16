from progressivis.core.utils import Dialog
from progressivis.core.slot import SlotDescriptor

from ..table.module import TableModule
from ..utils.psdict import PsDict
from collections import OrderedDict

class MergeDict(TableModule):
    """
    Binary join module to join two dict and return a third one.

    Slots:
        first : Table module producing the first dict to join
        second : Table module producing the second dict to join
    Args:
        kwds : argument to pass to the join function
    """
    def __init__(self, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('first', type=PsDict, required=True),
                         SlotDescriptor('second', type=PsDict, required=True)])
        super().__init__(**kwds)
        #self.join_kwds = self._filter_kwds(kwds, join)
        self._dialog = Dialog(self)

    async def run_step(self, run_number, step_size, howlong):
        first_slot = self.get_input_slot('first')
        first_slot.update(run_number)
        second_slot = self.get_input_slot('second')
        first_dict = first_slot.data()
        second_dict = second_slot.data()
        if first_dict is None or second_dict is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        second_slot.update(run_number)
        first_slot.created.next()
        second_slot.created.next()
        first_slot.updated.next()
        second_slot.updated.next()
        first_slot.deleted.next()
        second_slot.deleted.next()
        if self._table is None:
            self._table = PsDict(**first_dict, **second_dict)
        else:
            self._table.update(first_dict)
            self._table.update(second_dict)
        return self._return_run_step(self.next_state(first_slot), steps_run=1)
