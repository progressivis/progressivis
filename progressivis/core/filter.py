from progressivis.core.common import ProgressiveError, indices_len
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import pandas as pd

import logging
logger = logging.getLogger(__name__)

class Filter(DataFrameModule):
    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True),
                         SlotDescriptor('filter', type=pd.DataFrame, required=False)])
        super(Filter, self).__init__(**kwds)
        self.default_step_size = 1000
        
    def run_step(self,run_number,step_size,howlong):
        filter_slot = self.get_input_slot('filter')
        df_slot = self.get_input_slot('df')
        if not filter_slot:
            filter = None
        else:
            filter_df = filter_slot.data()
            filter_slot.update(run_number)
            if  filter_slot.has_created(): # ignore deleted and updated
                df_slot.reset() # re-filter
            indices = filter_slot.next_created() # read it all
            filter = self.last_row(filter_df)['filter'] # get the filter expression
            if filter is not None:
                filter = unicode(filter) # make sure we have a string

        if filter is None: # nothing to filter, just pass through
            logger.info('No filter, passing data through')
            self._df = df_slot.data()
            return self._return_run_step(self.state_blocked, steps_run=1)
        
        df_slot.update(run_number)
        if df_slot.has_deleted() or df_slot.has_updated():
            df_slot.reset()
            self._df = None
            df_slot.update(run_number)
        
        indices = df_slot.next_created(step_size)
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=steps)
        if isinstance(indices, slice):
            print indices
            indices = slice(indices.start, indices.stop-1)
        df = df_slot.data().loc[indices]
        try:
            filtered_df = df.eval(filter)
            if isinstance(filtered_df, pd.Series):
                filtered_df = df[filtered_df]
        except Exception as e:
            logger.error('Probably a syntax error in filter expression: %s', e)
            self._df = df_slot.data()
            return self._return_run_step(self.state_blocked, steps_run=steps)
        filtered_df[self.UPDATE_COLUMN] = run_number
        print 'len=%d/%d (step %d)' % (len(filtered_df), len(df_slot.data()), steps)
        if self._df is None:
            self._df = filtered_df
        else:
            self._df = self._df.append(filtered_df) # don't ignore index I think
        return self._return_run_step(self.state_blocked, steps_run=steps)
