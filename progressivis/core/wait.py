from progressivis.core.common import ProgressiveError
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

class Wait(DataFrameModule):
    parameters = [('delay', np.dtype(float), np.nan),
                  ('reads', np.dtype(int), -1)]

    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors', [SlotDescriptor('df', type=pd.DataFrame)])
        super(Wait, self).__init__(**kwds)
        if np.isnan(self.params.delay) and self.params.reads == -1:
            raise ProgressiveError('Module %s needs either a delay or a number of reads, not both',
                                   self.pretty_typename())
        
    def is_ready(self):
        if not super(Wait, self).is_ready():
            return False
        if self.is_zombie():
            return True # give it a chance to run before it dies
        delay = self.params.delay
        reads = self.params.reads
        if np.isnan(delay) and reads<0:
            return False
        inslot = self.get_input_slot('df')
        #if inslot.output_module is None: # should not happen, the slot is mandatory
        #    return False
        trace = inslot.output_module.tracer.df()
        if len(trace) == 0:
            return False
        if not np.isnan(delay):
            return len(trace) >= delay
        elif reads >= 0:
            return len(inslot.data()) >= reads
        return False

    def df(self):
        return self.get_input_slot('df').data()

    def predict_step_size(self, duration):
        return 1
    
    def run_step(self,run_number,step_size,howlong):
        return self._return_run_step(self.state_blocked, steps_run=1)
