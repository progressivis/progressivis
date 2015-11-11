# -*- coding: utf-8 -*-
""" Computes the distance matrix from each row of a data frame.
"""
from progressivis.core.common import ProgressiveError, indices_len
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances

import logging
logger = logging.getLogger(__name__)

class PairwiseDistances(DataFrameModule):
    def __init__(self, metric='euclidean', columns=None, n_jobs=1, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame),
                         SlotDescriptor('array', required=False)])
        super(PairwiseDistances, self).__init__(dataframe_slot='distance', **kwds)
        self.default_step_size = kwds.get('step_Size', 100)  # initial guess
        self._metric = metric
        self._n_jobs = n_jobs
        if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
            raise ProgressiveError('Unknown distance %s', metric)
        self.columns = columns

    def is_ready(self):
        if self.get_input_slot('df').has_created():
            return True
        return super(PairwiseDistances, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        df = dfslot.data()
        dfslot.update(run_number, df)
        if dfslot.has_updated() or dfslot.has_deleted():        
            dfslot.reset()
            logger.info('Reseting history because of changes in the input df')
            dfslot.update(run_number, df)
            #TODO: be smarter with changed values

        #len_b = (dfslot.last_index)
        #n = len(df)-len_b

        m = step_size
        
        indices = dfslot.next_created(m)
        m = indices_len(indices)

        i = None
        j = None
        Si = self._df

        arrayslot = self.get_input_slot('array')
        if arrayslot is not None and arrayslot.data() is not None:
            array = arrayslot.data()
            logger.info('Using array instead of DataFrame columns')
            if Si is not None:
                i = array[Si.index]
            j = array[indices]
        if j is None:
            if self.columns is None:
                self.columns = df.columns.delete(np.where(df.columns==DataFrameModule.UPDATE_COLUMN))
            elif not isinstance(self.columns, pd.Index):
                self.columns = pd.Index(self.columns)
            #row = None
            try:
                rows = df[self.columns]
            except Exception as e:
                logger.error('While extracting columns', e)
                raise
            if Si is not None:
                i = rows.loc[Si.index]
                assert len(i)==len(Si.index)
            j = rows.iloc[indices]
            assert len(j)==indices_len(indices)

        Sj = pairwise_distances(j, metric=self._metric, n_jobs=self._n_jobs)
        if Si is None:
            S = Sj
            index = dfslot.last_index[indices]
        else:
            Sij = pairwise_distances(i,j)
            Sji = Sij.T
            S1 = np.hstack((Si, Sij))
            S2 = np.hstack((Sji, Sj))
            S = np.vstack((S1, S2))
            index = Si.index.append(df.index[indices])
        self._df = pd.DataFrame(S,index=index)
        return self._return_run_step(dfslot.next_state(),
                                     steps_run=m, reads=m, updates=len(self._df))
