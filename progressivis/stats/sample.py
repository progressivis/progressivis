"""Simple and naive module to sample a progressive dataframe."""
from progressivis import *

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
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor('sample', type=pd.DataFrame, required=False)])
        super(Sample, self).__init__(**kwds)
         # probability associated with each sample. If n selected over N, p=n/N
        stickiness = self.params.stickiness
        if stickiness < 0 or stickiness >= 1:
            raise ProgressiveError('Invalid stickiness (%f) should be [0,1]', stickiness)
        self._df = None

    def predict_step_size(self, duration):
        # Module sample is constant time (supposedly)
        return 1

    def get_data(self, name):
        if name=='sample':
            return self.sample()
        return super(Sample,self).get_data(name)

    def df(self):
        return self.get_input_slot('df').data()

    def sample(self):
        return self._df

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        input_df = dfslot.data()
        l = len(input_df) if input_df is not None else 0

        if (l == 0):
            if self.df_ is not None and len(self._df) != 0:
                self._df = None
            self._return_run_step(self.state_blocked, steps_run=0)

        size = 1
        n = self.params.n
        frac = self.params.frac
        if n > 0:
            size = n
        elif not np.isnan(frac):
            size = np.max(0, math.floor(l*frac))

        self._df = input_df.sample(n=size)
        self._df.loc[:,self.UPDATE_COLUMN] = pd.Series(run_number, index=self._df.index)
        return self._return_run_step(self.state_blocked, steps_run=1, reads=l, updates=l)
