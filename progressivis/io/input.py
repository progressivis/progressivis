
import numpy as np

from progressivis.table import Table
from progressivis.table.module import TableModule


class Input(TableModule):
    parameters = [('history', np.dtype(int), 3)]
    schema = '{input: string}'

    def __init__(self, **kwds):
        super(Input, self).__init__(**kwds)
        self.tags.add(self.TAG_INPUT)
        self.result = Table(name=None, dshape=Input.schema, create=True)
        self._last = len(self.result)
        self.default_step_size = 1000000

    def is_ready(self):
        return len(self.result) > self._last

    def run_step(self, run_number, step_size, howlong):
        self._last = len(self.result)
        return self._return_run_step(self.state_blocked, steps_run=0)

    async def from_input(self, msg):
        if not isinstance(msg, (list, dict)):
            msg = {'input': msg}
        self.result.add(msg)
