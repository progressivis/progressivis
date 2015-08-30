from progressivis.core.common import ProgressiveError
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

from sklearn.manifold.mds import *

import logging
logger = logging.getLogger(__name__)

class MDS(DataFrameModule):
    schema = [('x', np.dtype(float), np.nan),
              ('y', np.dtype(float), np.nan),
              DataFrameModule.UPDATE_COLUMN_DESC]              

    schema = [('max_iter', np.dtype(int), 300),
              ('n_init', np.dtype(int), 8),
              ('eps', np.dtype(float), 1e-3),
              DataFrameModule.UPDATE_COLUMN_DESC]              

    def __init__(self, n_jobs=1, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('distance', type=pd.DataFrame)])
        super(MDS, self).__init__(dataframe_slot='mds', **kwds)
        self._step_size = 100
        self._n_jobs = n_jobs
        self._df = None
        self._cache = None

    def is_ready(self):
        if not (self.get_input_slot('distance').is_buffer_empty()):
            return True
        return super(MDS, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('distance')
        df = dfslot.data()
        dfslot.update(run_number, df)
        p = self.parameter
        n = len(df)

        if self._df is None:
            mds = MDS(2, max_iter=p.max_iter, dissimilarity='precomputed', n_init=p.n_init)
            if step_size >= n:
                index=df.index
            else:
                index=df.index[:step_size]
            S = df.loc[index,index]
            Y, stress = mds.fit_transform(S)
            # Ignore stress for now
            self._df = pd.DataFrame(Y,columns=['x', 'y'], index=index)
            self._df[self.UPDATE_COLUMN] = run_number
        else:
            m = len(self._df)
            if (m+step_size) >= n:
                index=df.index
            else:
                index=df.index[:m+step_size]
            S = df.loc[index,index]
            new_points = self.barycenter(self.df_, S, index)
            new_df = df.DataFrame(new_points, index=df.index[m:step_size])
            new_df[self.UPDATE_COLUMN] = run_number
            self.df_ = self._df.concat(new_df) # don't ignore index
            mds = MDS(2, max_iter=p.max_iter, dissimilarity="precomputed", n_init=1)
            Y, stress = mds.fit_transform(S, init=self.df_[['x','y']])
            self._df['x'] = Y[0]
            self._df['y'] = Y[1]
        next_state = state_ready if (len(self._df) < n) else state_blocked
        return self._return_run_step(next_state,
                                     steps_run=len(index), reads=len(index), updates=len(self._df))
            
