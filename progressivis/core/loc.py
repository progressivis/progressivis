from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor
from progressivis.core.synchronized import synchronized

import pandas as pd

import logging
logger = logging.getLogger(__name__)

class Loc(DataFrameModule):
    def __init__(self, indices, columns, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        super(Loc, self).__init__(**kwds)
        self._indices = indices
        self._columns = columns

    def predict_step_size(self, duration):
        return 1

    @synchronized
    def run_step(self,run_number,step_size,howlong):
        df_slot = self.get_input_slot('df')
        in_df = df_slot.data()
        if in_df is None:
            return self._return_run_step(self.state_blocked, 0)
        df_slot.update(run_number)
        try:
            self._df = in_df.loc[self._indices, self._columns]
        except Exception as e:
            logger.error('Cannot extract indices or columns: %s', e)
            self._df = None
        else:
            self._df[self.UPDATE_COLUMN] = run_number
        return self._return_run_step(self.state_blocked, steps_run=1)
