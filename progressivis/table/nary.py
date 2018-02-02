from __future__ import absolute_import, division, print_function

from progressivis.table.module import TableModule
from progressivis.table import Table
from progressivis.core.slot import SlotDescriptor


class NAry(TableModule):
    def __init__(self, nary='table', **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True)])
        super(NAry, self).__init__(**kwds)
        self.nary = nary
        self.inputs = [nary]
        
    def predict_step_size(self, duration):
        return 1

    def _add_input_slot(self, name):
        self.inputs.append(name)
        self.input_descriptors[name] = SlotDescriptor(name, type=Table, required=True)
        self._input_slots[name] = None

    # Magic input slot created 
    def _connect_input(self, slot):
        ret = self.get_input_slot(slot.input_name)
        if ret and slot.input_name==self.nary:
            name = '%s.%d' % (self.nary, len(self.inputs))
            self._add_input_slot(name)
            slot.input_name = name # patch the slot name
            ret = None
        self._input_slots[slot.input_name] = slot
        return ret
