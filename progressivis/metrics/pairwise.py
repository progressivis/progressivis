# -*- coding: utf-8 -*-
""" Computes the distance matrix from each row of a data frame.
"""
from progressivis.core.utils import ProgressiveError, indices_len
from progressivis.core.module import Module
from progressivis.core.buffered_matrix import BufferedMatrix
from progressivis.core.slot import SlotDescriptor
from progressivis.core.synchronized import synchronized


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances

import logging
logger = logging.getLogger(__name__)

class PairwiseDistances(Module):
    def __init__(self, metric='euclidean', columns=None, n_jobs=1, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame),
                         SlotDescriptor('array', required=False)])
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor('dist', type=np.ndarray, required=False)])
        super(PairwiseDistances, self).__init__(dataframe_slot='distance', **kwds)
        self.default_step_size = kwds.get('step_Size', 100)  # initial guess
        self.columns = columns
        self._metric = metric
        self._n_jobs = n_jobs
        self._last_index = None
        if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
            raise ProgressiveError('Unknown distance %s', metric)
        self._buf = BufferedMatrix()

    def is_ready(self):
        if self.get_input_slot('df').has_created():
            return True
        return super(PairwiseDistances, self).is_ready()

    def get_data(self, name):
        if name=='dist':
            return self._buf.matrix()
        return super(PairwiseDistances, self).get_data(name)

    @synchronized
    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        df = dfslot.data()
        dfslot.update(run_number)
        if dfslot.has_updated() or dfslot.has_deleted():        
            dfslot.reset()
            logger.info('Reseting history because of changes in the input df')
            dfslot.update(run_number, df)
            #TODO: be smarter with changed values

        m = step_size
        
        indices = dfslot.next_created(m)
        m = indices_len(indices)

        i = None
        j = None
        Si = self._buf.matrix()

        arrayslot = self.get_input_slot('array')
        if arrayslot is not None and arrayslot.data() is not None:
            array = arrayslot.data()
            logger.debug('Using array instead of DataFrame columns')
            if Si is not None:
                i = array[self._last_index]
            j = array[indices]
        if j is None:
            if self.columns is None:
                self.columns = df.columns.delete(np.where(df.columns==Module.UPDATE_COLUMN))
            elif not isinstance(self.columns, pd.Index):
                self.columns = pd.Index(self.columns)
            rows = df[self.columns]
            if Si is not None:
                i = rows.loc[self._last_index]
                assert len(i)==len(self._last_index)
            if isinstance(indices,slice):
                # pandas .loc idiosyncrasy. The doc says:
                # "note that contrary to usual python slices,
                #  both the start and the stop are included!"
                j = rows.loc[indices.start:indices.stop-1]
            else:
                j = rows.loc[indices]
            assert len(j)==indices_len(indices)

        Sj = pairwise_distances(j, metric=self._metric, n_jobs=self._n_jobs)
        if Si is None:
            mat = self._buf.resize(Sj.shape[0])
            mat[:,:] = Sj
            self._last_index = dfslot.last_index[indices]
        else:
            Sij = pairwise_distances(i,j)
            n0 = i.shape[0]
            n1 = n0+j.shape[0]
            mat = self._buf.resize(n1)
            mat[0:n0,n0:n1] = Sij
            mat[n0:n1,0:n0] = Sij.T
            mat[n0:n1,n0:n1] = Sj
            self._last_index = self._last_index.append(df.index[indices])
        return self._return_run_step(dfslot.next_state(), steps_run=m)
