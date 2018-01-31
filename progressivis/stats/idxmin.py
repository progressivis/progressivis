from __future__ import absolute_import, division, print_function

from progressivis.core.utils import indices_len, fix_loc
from progressivis.core.slot import SlotDescriptor
from progressivis.core.synchronized import synchronized
from progressivis.table.module import TableModule
from progressivis.table.table import Table
from collections import OrderedDict
import numpy as np
import logging
logger = logging.getLogger(__name__)


class IdxMin(TableModule):
    parameters = [('history', np.dtype(int), 3)]

    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True)])
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor('min', type=Table, required=False)])
        super(IdxMin, self).__init__(**kwds)
        self._min = None
        self.default_step_size = 10000

    def min(self):
        return self._min

    def get_data(self, name):
        if name=='min':
            return self.min()
        return super(IdxMin,self).get_data(name)

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super(IdxMin, self).is_ready()

    @synchronized
    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('table')
        dfslot.update(run_number, self.id)
        if dfslot.updated.any() or dfslot.deleted.any():        
            dfslot.reset(mid=self.id)
            self._df = None
            dfslot.update(run_number, self.id)
        indices = dfslot.created.next(step_size) # returns a slice
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        input_df = dfslot.data()
        op = self.filter_columns(input_df, fix_loc(indices)).idxmin()
        #if not op.index.equals(self._columns):
        #    # some columns are not numerical
        #    self._columns = op.index


        if self._min is None:
            min_ = OrderedDict(zip(op.keys(), [np.nan]*len(op.keys())))
            for col, ix in op.items():
                min_[col] = input_df.at[ix, col] # lookup value, is there a better way?
            self._min = Table(self.generate_table_name('_min'),
                                  dshape=input_df.dshape,
#                                  scheduler=self.scheduler(),
                                  create=True)
            self._min.append(min_, indices=[run_number])
            self._df = Table(self.generate_table_name('_df'),
                                 dshape=input_df.dshape,
#                                 scheduler=self.scheduler(),
                                 create=True)
            self._df.append(op, indices=[run_number])
        else:
            prev_min = self._min.last()
            prev_idx = self._df.last()
            min_ = OrderedDict(prev_min.items()) 
            for col, ix in op.items():
                val = input_df.at[ix, col]
                if np.isnan(val):
                    pass
                elif np.isnan(min_[col]) or val < min_[col]:
                    op[col] = prev_idx[col]
                    min_[col] = val
            with self.lock:
                self._df.append(op, indices=[run_number])
                self._min.append(min_, indices=[run_number])
                if len(self._df) > self.params.history:
                    self._df = Table(self.generate_table_name('_df'),
                                         data=self._df.loc[self._df.index[-self.params.history:]],
#                                         scheduler=self.scheduler(),
                                         create=True)
                    self._min = Table(self.generate_table_name('_min'),
                                          data=self._min.loc[self._min.index[-self.params.history:]],
#                                          scheduler=self.scheduler(),
                                          create=True)

        return self._return_run_step(self.next_state(dfslot), steps_run=steps)
