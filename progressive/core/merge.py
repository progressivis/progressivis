from progressive.core.common import ProgressiveError
from progressive import DataFrameModule, SlotDescriptor

import numpy as np

class Merge(DataFrameModule):
    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('in', type=pd.DataFrame, required=True)])
        super(Merge, self).__init__(**kwds)
        self.inputs = ['in']
        
    def predict_step_size(self, duration):
        return 1

    def _add_input_slot(self, name):
        self.inputs.append(name)
        self.input_descriptors.append(SlotDescriptor(name, type=pd.DataFrame, required=False))
        self._input_slots[slot] = None

    # Magic input slot created 
    def _connect_input(self, slot):
        ret = self.get_input_slot(slot.input_name)
        if ret and slot.input_name=='in':
            name = 'in.%d' % len(self.inputs)
            self._add_input_slot(name)
            slot.input_name = name # patch the slot name
            ret = None
        self._input_slots[slot.input_name] = slot
        return ret
    
    def run_step(self, step_size, howlong):
        frames = [ self.get_input_slot(name).data() for name in self.inputs]
        self._df = concat(frames, axis=1, join_axes=[frames[0].index])
        self._return_run_step(self.state_blocked, steps_run=1, reads=len(self._df), updates=len(self._df))
