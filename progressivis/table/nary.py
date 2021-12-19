"Base Module for modules supporting a variable number in input slots."

from progressivis.table.module import TableModule
from progressivis.table import BaseTable
from progressivis.core.slot import SlotDescriptor


class NAry(TableModule):
    "Base class for modules supporting a variable number of input slots."
    inputs = [SlotDescriptor("table", type=BaseTable, required=True, multiple=True)]

    def __init__(self, nary="table", **kwds):
        super(NAry, self).__init__(**kwds)
        self.nary = nary
        self.inputs = [nary]

    def predict_step_size(self, duration):
        return 1

    def get_input_slot_multiple(self, name=None):
        if name is None:
            name = self.nary
        return super(NAry, self).get_input_slot_multiple(name)

    # def _add_input_slot(self, name):
    #     self.inputs.append(name)
    #     self.input_descriptors[name] = SlotDescriptor(name, type=BaseTable)
    #     self._input_slots[name] = None

    # # Magic input slot created
    # def _connect_input(self, slot):
    #     ret = self.get_input_slot(slot.input_name)
    #     if ret and slot.input_name == self.nary:
    #         name = '%s.%d' % (self.nary, len(self.inputs))
    #         self._add_input_slot(name)
    #         slot.input_name = name  # patch the slot name
    #         ret = None
    #     self._input_slots[slot.input_name] = slot
    #     return ret

    def run_step(self, run_number, step_size, howlong):  # pragma no cover
        raise NotImplementedError("run_step not defined")
