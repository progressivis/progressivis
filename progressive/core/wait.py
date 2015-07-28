from progressive.core.common import ProgressiveError
from progressive import Module, SlotDescriptor

import numpy as np

class Wait(Module):
    parameters = [('delay', np.dtype(float), np.nan),
                  ('reads', np.dtype(int), 0)]

    def __init__(self, delay=None, reads=None, **kwds):
        if delay is None and reads is None:
            raise ProgressiveError('Module %s needs a delay or a number of reads', self.__class__.name)
        if delay is not None and reads is not None:
            raise ProgressiveError('Module %s needs either a delay or a number of reads', self.__class__.name)
        self._add_slots(kwds,'output_descriptors', [SlotDescriptor('out')])
        self._add_slots(kwds,'input_descriptors', [SlotDescriptor('in')])
        super(Wait, self).__init__(**kwds)
        self.delay = delay
        self.reads = reads
        
    def is_ready(self):
        inslot = self.get_input_slot('in')
        if inslot.output_module is None:
            return False
        trace = inslot.output_module.tracer.df() 
        if self.delay:
            return trace[Module.UPDATE_COLUMN].irow(-1) >= self.delay
        elif self.reads:
            return trace['reads'].irow(-1) >= self.reads
        return False

    def predict_step_size(self, duration):
        return 1
    
    def run_step(self, step_size, howlong):
        print 'running wait'
        raise StopIteration()
