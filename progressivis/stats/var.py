from ..core.utils import indices_len, fix_loc
from ..core.slot import SlotDescriptor
from ..table.module import TableModule
from ..table.table import Table
from ..table.dshape import dshape_all_dtype
from ..utils.psdict import PsDict
from ..core.decorators import process_slot, run_if_any


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
        if np.isnan(datum):
            return
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)
        n_ddof = self.n - self.ddof
        self.variance = self.M2 / n_ddof if n_ddof else np.nan

    @property
    def std(self):
        return np.sqrt(self.variance)


class VarH(TableModule):
    """
    Compute the variance of the columns of an input table.
    """
    parameters = [('history', np.dtype(int), 3)]
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, columns=None, **kwds):
        super().__init__(dataframe_slot='table', **kwds)
        self._columns = columns
        self._data = {}
        self.default_step_size = 1000

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super().is_ready()

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
        if self.result is None:
            self.result.resize(0)

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
                ds = dshape_all_dtype(input_df.columns, np.dtype("float64"))
                self.result = Table(self.generate_table_name('var'),
                                    dshape=ds,  # input_df.dshape,
                                    create=True)
            self.result.append(op, indices=[run_number])
            return self._return_run_step(self.next_state(dfslot), steps)


class Var(TableModule):
    """
    Compute the variance of the columns of an input table.
    This variant didn't keep history
    """
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, columns=None, **kwds):
        super().__init__(dataframe_slot='table', **kwds)
        self._columns = columns
        self._data = {}
        self.default_step_size = 1000

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super().is_ready()

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
        if self.result is None:
            self.result.resize(0)

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
                self.result = PsDict(op)
            else:
                self.result.update(op)
            return self._return_run_step(self.next_state(dfslot), steps)
