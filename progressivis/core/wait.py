from progressivis.core.common import ProgressiveError
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

class Wait(DataFrameModule):
    parameters = [('delay', np.dtype(float), np.nan),
                  ('reads', np.dtype(int), 0)]

    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors', [SlotDescriptor('inp', type=pd.DataFrame)])
        super(Wait, self).__init__(dataframe_slot='out', **kwds)
        
    def is_ready(self):
        if not super(Wait, self).is_ready():
            return False
        delay = self.params.delay
        reads = self.params.reads
        if delay==np.nan and reads==0:
            return False
        if delay!=np.nan and reads != 0:
            raise ProgressiveError('Module %s needs either a delay or a number of reads, not both', self.__class__.__name__)
        inslot = self.get_input_slot('inp')
        if inslot.output_module is None:
            return False
        trace = inslot.output_module.tracer.df()
        if len(trace) == 0:
            return False
        if delay != np.nan:
            return len(trace) >= delay
        elif reads:
            return trace['reads'].irow(-1) >= reads
        return False

    def get_data(self, name):
        if name=='out': # passes input slot through
            inslot = self.get_input_slot('inp')
            if inslot:
                return inslot.data()
        return super(Wait, self).get_data(name)

    def predict_step_size(self, duration):
        return 1
    
    def run_step(self,run_number,step_size,howlong):
        return self._return_run_step(self.state_blocked, steps_run=1)
