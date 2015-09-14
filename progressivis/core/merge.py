from progressivis.core.common import ProgressiveError
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import pandas as pd
import numpy as np

class Merge(DataFrameModule):
    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        super(Merge, self).__init__(**kwds)
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
    
    def run_step(self,run_number,step_size,howlong):
        frames = []
        for name in self.inputs:
            df = self.get_input_slot(name).data()
            df = df[df.columns.difference([self.UPDATE_COLUMN])]
            frames.append(df)
        self._df = pd.concat(frames, axis=1, join_axes=[frames[0].index])
        self._df[self.UPDATE_COLUMN] = run_number
        return self._return_run_step(self.state_blocked, steps_run=1, reads=len(self._df), updates=len(self._df))
