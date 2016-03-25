from progressivis.core.utils import indices_len, fix_loc
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor
from progressivis.core.synchronized import synchronized

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

# Should translate that to Cython eventually
class OnlineVariance(object):
    """
    Welford's algorithm computes the sample variance incrementally.
    """

    def __init__(self, ddof=1):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0

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


class Var(DataFrameModule):
    """
    Compute the variance of the columns of an input dataframe.
    """
    parameters = [('history', np.dtype(int), 3)]

    def __init__(self, columns=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        super(Var, self).__init__(**kwds)
        self._columns = columns
        self._data = {}
        self.default_step_size = 1000

    def is_ready(self):
        if self.get_input_slot('df').has_created():
            return True
        return super(Var, self).is_ready()

    def op(self, chunk):
        cols = chunk.columns
        ret = []
        for c in cols:
            data = self._data.get(c)
            if data is None:
                data = OnlineVariance()
                self._data[c] = data
            data.add(chunk[c])
            ret.append(data.variance)
        return pd.Series(ret, index=cols)

    @synchronized
    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)
        if dfslot.has_updated() or dfslot.has_deleted():        
            dfslot.reset()
            self._df = None
            dfslot.update(run_number)
        indices = dfslot.next_created(step_size) # returns a slice
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        input_df = dfslot.data()
        op = self.op(self.filter_columns(input_df,fix_loc(indices)))
        op[self.UPDATE_COLUMN] = run_number
        if self._df is None:
            self._df = pd.DataFrame([op], index=[run_number])
        else:
            self._df.loc[run_number] = op
        print self._df

        if len(self._df) > self.params.history:
            self._df = self._df.loc[self._df.index[-self.params.history:]]
        return self._return_run_step(dfslot.next_state(), steps_run=steps)
