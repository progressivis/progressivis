"""Simple and naive module to sample a progressive dataframe."""
from progressivis import DataFrameModule, SlotDescriptor, ProgressiveError

import numpy as np
import pandas as pd


import logging
logger = logging.getLogger(__name__)

class Sample(DataFrameModule):
    parameters = [('n',  np.dtype(int),   100),
                  ('frac',   np.dtype(float), np.nan),
                  ('stickiness',   np.dtype(float), 0.9)] # probability of a sample to remain

    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame)])
        super(Sample, self).__init__(**kwds)
         # probability associated with each sample. If n selected over N, p=n/N
        stickiness = self.params.stickiness
        if stickiness < 0 or stickiness >= 1:
            raise ProgressiveError('Invalid stickiness (%f) should be [0,1]', stickiness)
        self._df = None

    def predict_step_size(self, duration):
        # Module sample is constant time (supposedly)
        return 1

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)
        # do not produce another sample is nothing has changed
        if not (dfslot.has_created() or dfslot.has_deleted() or dfslot.has_updated()):
            return self._return_run_step(self.state_blocked, steps_run=0)
        if dfslot.has_deleted() or dfslot.has_updated():
            dfslot.reset()
            dfslot.update(run_number)
            self._df = None
        input_df = dfslot.data()
        l = len(input_df) if input_df is not None else 0

        if (l == 0):
            if self._df is not None and len(self._df) != 0:
                self._df = None
            self._return_run_step(self.state_blocked, steps_run=0)

        size = 1
        n = self.params.n
        frac = self.params.frac
        if n > 0:
            size = n
        elif not np.isnan(frac):
            size = np.max(0, np.floor(l*frac))
        if size >= len(input_df):
            self._df = input_df
        else:
            self._df = input_df.sample(n=size)
            self._df.loc[:,self.UPDATE_COLUMN] = run_number
        return self._return_run_step(self.state_blocked, steps_run=1, reads=l, updates=l)
