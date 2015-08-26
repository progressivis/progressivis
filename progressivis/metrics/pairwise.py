# -*- coding: utf-8 -*-
""" Computes the distance matrix from each row of a data frame.
"""
from progressivis.core.common import ProgressiveError
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances

import logging
logger = logging.getLogger(__name__)

class PairwiseDistances(DataFrameModule):
    def __init__(self, metric='euclidean', columns=None, n_jobs=1, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame)])
        super(PairwiseDistances, self).__init__(**kwds)
        self._metric = metric
        self._n_jobs = n_jobs
        if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
            raise ProgressiveError('Unknown distance %s', metric)
        self.columns = columns

    def is_ready(self):
        if not (self.get_input_slot('df').is_buffer_empty()):
            return True
        return super(PairwiseDistances, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        df = dfslot.data()
        dfslot.update(run_number, df)
        if len(dfslot.deleted) or len(dfslot.updated) > len(dfslot.created):
            dfslot.reset()
            logger.info('Reseting history because of changes in the input df')
            #TODO: be smarter with changed values
        dfslot.buffer_created()

        len_b = len(dfslot.created)
        n = len(df)-len_b

        if self.columns is None:
            self.columns = df.columns.delete(np.where(df.columns==DataFrameModule.UPDATE_COLUMN))
        elif not isinstance(self.columns, pd.Index):
            self.columns = pd.Index(self.columns)

        row = None
        try:
            rows = df[self.columns]
        except Exception as e:
            logger.error('While extracting columns', e)
            raise

        # We have the old matrix Si of size (n), we want to complete it with
        # two sub matrices, Sij and Sj of length (m).
        # We are given a "budget" of step_size (s) operations
        # See how many new rows we can compute with our budget.
        # These will be the (j) new rows
        # We need to comput Sij, which will take n*m operations
        # and Sj which will take m*m, so we search m for n*m + m*m = s
        # m^2 + n*m -s = 0, a=1, b=n, c=-s, solution is -b +- sqrt(b^2-4ac)/2a
        # The only positive solution is -n + sqrt(n^2+4s) / 2

        m = (-n + np.sqrt(n*n + 4*step_size)) / 2.0
        m = np.min([1.0, m])

        Si = self._df
        indices = dfslot.next_buffered(m)
        j = rows[indices]
        Sj = pairwise_distances(j, metric=self.metric, n_jobs=self.n_jobs)
        if Si is None:
            S = Sj
            index = Si.index
        else:
            i = df[Si.index]
            Sij = pairwise_distances(i,j)
            Sji = Sij.T
            S1 = np.hstack((Si, Sij))
            S2 = np.hstack((Sji, Sj))
            S = np.vstack((S1, S2))
            index = Si.index + df.index[indices]
        self._df = pd.DataFrame(S,index=index)
