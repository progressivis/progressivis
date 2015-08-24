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

    schema = [('selected', np.dtype(bool), False),
              DataFrameModule.UPDATE_COLUMN_DESC]              

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
        self._sample = self.create_dataframe(Sample.schema)
        self._cache = None

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
        input_df = self.df()
        if self._cache is None:
            self._cache = input_df[self._sample['selected']]
        return self._cache

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        input_df = dfslot.data()
        l = len(input_df) if input_df is not None else 0

        if (l == 0):
            if self.df_ is not None and len(self._df) != 0:
                self._previous_choice = NIL
                self._df = self.create_dataframe(Histogram2D.schema)
            self._return_run_step(self.state_blocked, steps_run=1)

        size = 1
        n = self.params.n
        frac = self.params.frac
        if n > 0:
            size = n
        elif not np.isnan(frac):
            size = np.max(0, math.floor(l*frac))

        if size >= l:
            logger.info('Sample returning the whole dataframe')
            self._sample = pd.Dataframe({'selected': True,
                                         self.UPDATE_COLUMN: run_number},
                                         index=input_df.index);
            self._cache = None
            return self._return_run_step(self.state_blocked, steps_run=1, reads=l, updates=l)

        #import pdb
        #pdb.set_trace()
        self._sample = self._sample.reindex(input_df.index)
        selected = self._sample['selected']
        selected.fillna(False, inplace=True)
        updates = self._sample[self.UPDATE_COLUMN]
        updates.fillna(run_number, inplace=True)
        weights = np.empty(l,dtype=float)
        std_p = 1.0 / l
        weights.fill(std_p)
        # already selected indices have weights specified by stickiness (usually high)
        weights[(selected==True).values] = np.max(self.params.stickiness, std_p)
        # other points have standard weight/probability
        weights = weights / weights.sum() # normalize weights
        locs = np.random.choice(l, size=size, replace=False, p=weights)
        locs.sort()
        # now, change the timestamp for values that changed
        new_selected = pd.Series(False,index=input_df.index)
        new_selected[locs] = True
        self._sample.loc[selected != new_selected, self.UPDATE_COLUMN] = run_number
        self._sample['selected'] = new_selected
        self._cache = None
        return self._return_run_step(self.state_blocked, steps_run=1, reads=l, updates=l)
