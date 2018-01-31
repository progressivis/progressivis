from __future__ import absolute_import, division, print_function

import numpy as np

from progressivis.table import Table
from progressivis.table.module import TableModule

class Input(TableModule):
    parameters = [('history', np.dtype(int), 3)]
    schema = '{input: string}'

    def __init__(self, **kwds):
        super(Input, self).__init__(**kwds)
        self._table = Table(name=None, dshape=Input.schema, create=True)
        self._last = len(self._table)
        self.default_step_size = 1000000


    def is_ready(self):
        return len(self._table) > self._last

    def run_step(self,run_number,step_size, howlong):
        self._last = len(self._table)
        return self._return_run_step(self.state_blocked, steps_run=0)
        
    def from_input(self, msg):
        if not isinstance(msg, (list, dict)):
            msg = {'input': msg}
        self._table.add(msg)

    def is_input(self):
        return True
