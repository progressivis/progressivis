from progressivis.core.common import ProgressiveError
from progressivis.core.module import Module
from progressivis.core.slot import SlotDescriptor

import numpy as np

class Wait(Module):
    parameters = [('delay', np.dtype(float), np.nan),
                  ('reads', np.dtype(int), 0)]

    def __init__(self, **kwds):
        self._add_slots(kwds,'output_descriptors', [SlotDescriptor('out')])
        self._add_slots(kwds,'input_descriptors', [SlotDescriptor('inp')])
        super(Wait, self).__init__(**kwds)
        
    def is_ready(self):
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
            return trace['end'].irow(-1) >= delay
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
        #print 'running wait'
        raise StopIteration()
