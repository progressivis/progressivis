from progressivis.core.common import ProgressiveError
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd
from sklean.metrics.pairwise import _VALID_METRICS

import logging
logger = logging.getLogger(__name__)

class PairedDistances(DataFrameModule):
    def __init__(self, metric='euclidean', n_jobs=1, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df1', type=pd.DataFrame)],
                        [SlotDescriptor('df2', type=pd.DataFrame)])
        super(PairedDistances, self).__init__(**kwds)
        self._metric = metric
        self._n_jobs = n_jobs
        if (metric not in _VALID_METRICS and
            not callable(metric) and metric != "precomputed"):
            raise ProgressiveError('Unknown distance %s', metric)
            

    def is_ready(self):
        if not (self.get_input_slot('df1').is_buffer_empty() or
                self.get_input_slot('df2').is_buffer_empty()):
            return True
        return super(PairedDistances, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        df1slot = self.get_input_slot('df1')
        df1 = df1slot.data()
        df1slot.update(run_number, df1)
        if len(df1slot.deleted) or len(df1slot.updated) > len(df1slot.created):
            df1slot.reset()
            logger.info('Reseting history because of changes in the input df1')
        df1slot.buffer_created()

        df2slot = self.get_input_slot('df2')
        if df2slot.output_module == df1slot.output_module and
            df2slot.output_name == df1slot.output_name:
            df2slot = df1slot
            df2 = df1
        else:
            df2 = df2slot.data()
            df2slot.update(run_number, df2)
            if len(df2slot.deleted) or len(df2slot.updated) > len(df2slot.created):
                df2slot.reset()
                logger.info('Reseting history because of changes in the input df2')
            df2slot.buffer_created()
        
        #TODO not finished
        
        
