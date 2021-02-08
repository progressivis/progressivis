from ..core.utils import indices_len, fix_loc
from ..core.slot import SlotDescriptor
from ..table.module import TableModule
from ..table.table import Table
from ..core.decorators import *


import numpy as np

import logging
logger = logging.getLogger(__name__)


# Should translate that to Cython eventually
class OnlineVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        self.delta = 0
        self.variance = 0

    def add(self, iterable):
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum):
        if np.isnan(datum): return
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)
        self.variance = self.M2 / (self.n - self.ddof)

    @property
    def std(self):
        return np.sqrt(self.variance)


class Var(TableModule):
    """
    Compute the variance of the columns of an input dataframe.
    """
    parameters = [('history', np.dtype(int), 3)]
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, columns=None, **kwds):
        super(Var, self).__init__(dataframe_slot='table', **kwds)
        self._columns = columns
        self._data = {}
        self.default_step_size = 1000

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super(Var, self).is_ready()

    def op(self, chunk):
        cols = chunk.columns
        ret = {}
        for c in cols:
            data = self._data.get(c)
            if data is None:
                data = OnlineVariance()
                self._data[c] = data
            data.add(chunk[c])
            ret[c] = data.variance
        return ret

    def reset(self):
        self.result = None

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            dfslot = ctx.table        
            indices = dfslot.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            op = self.op(self.filter_columns(input_df, fix_loc(indices)))
            if self.result is None:
                self.result = Table(self.generate_table_name('var'),
                                    dshape=input_df.dshape,
                                    create=True)
            self.result.append(op, indices=[run_number])
            print(self.result)

            if len(self.result) > self.params.history:
                self.result = self.result.loc[self.result.index[-self.params.history:]]
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)
