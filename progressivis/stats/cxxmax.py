from __future__ import absolute_import, division, print_function

from progressivis.table.module import TableModule
from progressivis.table.table import Table
from progressivis.core.slot import SlotDescriptor
try:
    from .progressivis_max import Max as CxxMax
except:
    CxxMax = None

import numpy as np

import logging
logger = logging.getLogger(__name__)


class Max(TableModule):
    parameters = [('history', np.dtype(int), 3)]

    def __init__(self, columns=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True)])
        super(Max, self).__init__(**kwds)
        self._columns = columns
        self.default_step_size = 10000
        self.cxx_module = CxxMax(self)

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super(Max, self).is_ready()

    def run_step(self, run_number, step_size, howlong):
        return self.cxx_module.run(run_number, step_size, howlong)
