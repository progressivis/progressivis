from progressivis.core.utils import ProgressiveError, indices_len, fix_loc, last_row
from progressivis.core.dataframe import DataFrameModule
from .slot import SlotDescriptor
from .utils import is_valid_identifier
from .buffered_dataframe import BufferedDataFrame
from sklearn.utils.extmath import row_norms

import pandas as pd

import logging
logger = logging.getLogger(__name__)

class SelectDelta(DataFrameModule):
    """
    Propagate changes only if their magnitude is above some threshold

    The threshold or ``delta'' can be absolute or relative if a scale column is supplied.
    """
    def __init__(self, delta=0, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True),
                         SlotDescriptor('scale', type=pd.DataFrame, required=False)])
        super(SelectDelta, self).__init__(**kwds)
        if delta < 0:
            raise ProgressiveError('delta (%s) should be positive or 0', delta)
        self._delta = delta
        self.default_step_size = 1000
        self._buffer = BufferedDataFrame()

    def reset(self):
        self._buffer = BufferedDataFrame()
        self._df = None

    def get_scale(self):
        scale_slot = self.get_input_slot('scale')
        if scale_slot is None:
            return 1
        scale_df = scale_slot.data()
        if scale_df is None or len(scale_df)==0:
            return 1
        return last_row(scale_df).iloc[0]

    def run_step(self,run_number,step_size,howlong):
        df_slot = self.get_input_slot('df')
        df_slot.update(run_number, buffer_created=True, buffer_updated=True)
        if df_slot.has_deleted():
            self.reset()
            df_slot.reset()
            df_slot.update(run_number)
        input_df = df_slot.data()
        columns = self.get_columns(input_df)
        if input_df is None or len(input_df)==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        indices = df_slot.next_created(step_size)
        steps = indices_len(indices)
        step_size -= steps
        steps_run = steps
        if steps != 0:
            indices = fix_loc(indices)
            self._buffer.append(input_df.loc[indices])
            self._df = self._buffer.df()
            self._df.loc[indices,self.UPDATE_COLUMN] = run_number
        if step_size > 0 and df_slot.has_updated():
            indices = df_slot.next_updated(step_size,as_slice=False)
            steps = indices_len(indices)
            if steps != 0:
                steps_run += steps
                indices = fix_loc(indices) # no need, but stick to the stereotype
                updated = self.filter_columns(input_df, indices)
                df = self.filter_columns(self._df, indices)
                norms = row_norms(updated-df)
                selected = (norms > (self._delta*self.get_scale()))
                indices = indices[selected]
                if selected.any():
                    logger.debug('updating at %d', run_number)
                    self._df.loc[indices, self._columns] = updated.loc[indices, self._columns]
                    self._df.loc[indices, self.UPDATE_COLUMN] = run_number
                else:
                    logger.debug('Not updating at %d', run_number)
        return self._return_run_step(df_slot.next_state(), steps_run=steps_run)

