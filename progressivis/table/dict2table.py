from progressivis.core.slot import SlotDescriptor
from .module import TableModule
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
    inputs = [SlotDescriptor('dict_', type=PsDict, required=True)]

    def __init__(self, history=False, **kwds):
        super().__init__(**kwds)

    def run_step(self, run_number, step_size, howlong):
        dict_slot = self.get_input_slot('dict_')
        # dict_slot.update(run_number)
        dict_ = dict_slot.data()
        if dict_ is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if not (dict_slot.created.any() or
                dict_slot.updated.any() or
                dict_slot.deleted.any()):
            return self._return_run_step(self.state_blocked, steps_run=0)
        dict_slot.created.next()
        dict_slot.updated.next()
        dict_slot.deleted.next()
        if self._table is None:
            self._table = Table(name=None, dshape=dict_.dshape)
        if len(self._table) == 0:  # or history:
            self._table.append(dict_.as_row)
        else:
            self._table.loc[0] = dict_.array
        return self._return_run_step(self.next_state(dict_slot), steps_run=1)
