from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import pandas as pd

class NAry(DataFrameModule):
    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        super(NAry, self).__init__(**kwds)
        self.inputs = ['df']
        
    def predict_step_size(self, duration):
        return 1

    def _add_input_slot(self, name):
        self.inputs.append(name)
        self.input_descriptors[name] = SlotDescriptor(name, type=pd.DataFrame, required=True)
        self._input_slots[name] = None

    # Magic input slot created 
    def _connect_input(self, slot):
        ret = self.get_input_slot(slot.input_name)
        if ret and slot.input_name=='df':
            name = 'df.%d' % len(self.inputs)
            self._add_input_slot(name)
            slot.input_name = name # patch the slot name
            ret = None
        self._input_slots[slot.input_name] = slot
        return ret

