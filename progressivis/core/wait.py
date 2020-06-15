
from progressivis.utils.errors import ProgressiveError
from progressivis.core.module import Module
from progressivis.core.slot import SlotDescriptor

import numpy as np


class Wait(Module):
    parameters = [('delay', np.dtype(float), np.nan),
                  ('reads', np.dtype(int), -1)]

    def __init__(self, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('inp', required=True)])
        self._add_slots(kwds, 'output_descriptors',
                        [SlotDescriptor('out', required=False)])
        super(Wait, self).__init__(**kwds)
        if np.isnan(self.params.delay) and self.params.reads == -1:
            raise ProgressiveError('Module %s needs either a delay or '
                                   'a number of reads, not both',
                                   self.pretty_typename())

    def is_ready(self):
        if not super(Wait, self).is_ready():
            return False
        if self.is_zombie():
            return True  # give it a chance to run before it dies
        delay = self.params.delay
        reads = self.params.reads
        if np.isnan(delay) and reads < 0:
            return False
        inslot = self.get_input_slot('inp')
        trace = inslot.output_module.tracer.trace_stats()
        if len(trace) == 0:
            return False
        if not np.isnan(delay):
            return len(trace) >= delay
        elif reads >= 0:
            return len(inslot.data()) >= reads
        return False

    def get_data(self, name):
        if name == 'inp':
            return self.get_input_slot('inp').data()
        return None

    def predict_step_size(self, duration):
        return 1

    def run_step(self, run_number, step_size, howlong):
        return self._return_run_step(self.state_blocked, steps_run=1)
