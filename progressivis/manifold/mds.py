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

    def __init__(self, n_jobs=1, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('distance', type=pd.DataFrame)])
        super(MDS, self).__init__(dataframe_slot='mds', **kwds)
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
        if len(dfslot.deleted) or len(dfslot.updated) > len(dfslot.created):
            dfslot.reset()
            logger.info('Reseting history because of changes in the input df')
        dfslot.buffer_created()
        
            
    
