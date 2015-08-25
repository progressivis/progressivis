# -*- coding: utf-8 -*-
""" Computes the distance matrix from each row of a data frame.

The matrix is kept as a `condensed distance matrix` form.
See http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html

"""
from progressivis.core.common import ProgressiveError
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd
from spipy.spatial.distance import pdist
from sklean.metrics.pairwise import _VALID_METRICS, pairwise_distances

import logging
logger = logging.getLogger(__name__)

class PairwiseDistances(DataFrameModule):
    schema = [('distance', np.dtype(float), np.nan),
              DataFrameModule.UPDATE_COLUMN_DESC]

    def __init__(self, metric='euclidean', columns=None, n_jobs=1, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame)])
        super(PairedDistances, self).__init__(**kwds)
        self._metric = metric
        self._n_jobs = n_jobs
        if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
            raise ProgressiveError('Unknown distance %s', metric)
        self.columns = columns
        self._df = self.create_dataframe(PairedDistances.schema)

    def is_ready(self):
        if not (self.get_input_slot('df').is_buffer_empty()):
            return True
        return super(PairedDistances, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        df = dfslot.data()
        dfslot.update(run_number, df)
        if len(dfslot.deleted) or len(dfslot.updated) > len(dfslot.created):
            dfslot.reset()
            logger.info('Reseting history because of changes in the input df')
        dfslot.buffer_created()

        len_b = len(dfslot.created)
        len_a = len(df)-len_b

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

        # We have the old matrix Si, we want to complete it with
        # two sub matrices, Sij and Sj.
        # We are given a "budget" of step_size operations
        # See how many new rows fit. These will be the (j) new rows

        if len_a == 0:
            steps = len_b*(len_b-1)/2
            if steps <= step_size:
                self._df['distance'] = pdist(rows)
                self._df[DataFrameModule.UPDATE_COLUMN] = run_number
            
        else:
            
