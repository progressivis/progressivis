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


class IdxMax(TableModule):
    parameters = [('history', np.dtype(int), 3)]

    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True)])
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor('max', type=Table, required=False)])
        super(IdxMax, self).__init__(**kwds)
        self._max = None
        self.default_step_size = 10000

    def max(self):
        return self._max

    def get_data(self, name):
        if name=='max':
            return self.max()
        return super(IdxMax,self).get_data(name)

    def is_ready(self):
        if self.get_input_slot('table').created.any():
            return True
        return super(IdxMax, self).is_ready()

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
        op = self.filter_columns(input_df, fix_loc(indices)).idxmax()

        if self._max is None:
            max_ = OrderedDict(zip(op.keys(), [np.nan]*len(op.keys())))
            for col, ix in op.items():
                max_[col] = input_df.at[ix, col] # lookup value, is there a better way?
            self._max = Table(self.generate_table_name('_max'),
                                  dshape=input_df.dshape,
#                                  scheduler=self.scheduler(),
                                  create=True)
            self._max.append(max_, indices=[run_number])
            self._df = Table(self.generate_table_name('_df'),
                                 dshape=input_df.dshape,
#                                 scheduler=self.scheduler(),
                                 create=True)
            self._df.append(op, indices=[run_number])
        else:
            prev_max = self._max.last()
            prev_idx = self._df.last()
            max_ = OrderedDict(prev_max.items()) 
            for col, ix in op.items():
                val = input_df.at[ix, col]
                if np.isnan(val):
                    pass
                elif np.isnan(max_[col]) or val > max_[col]:
                    op[col] = prev_idx[col]
                    max_[col] = val
            with self.lock:
                self._df.append(op, indices=[run_number])
                self._max.append(max_, indices=[run_number])
                if len(self._df) > self.params.history:
                    self._df = Table(self.generate_table_name('_df'),
                                         data=self._df.loc[self._df.index[-self.params.history:]],
#                                         scheduler=self.scheduler(),
                                         create=True)
                    self._max = Table(self.generate_table_name('_max'),
                                          data=self._max.loc[self._max.index[-self.params.history:]],
#                                          scheduler=self.scheduler(),
                                          create=True)
        return self._return_run_step(self.next_state(dfslot), steps_run=steps)
