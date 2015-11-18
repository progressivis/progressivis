from progressivis.core.common import ProgressiveError, indices_len
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor
from .utils import is_valid_identifier

import pandas as pd

import logging
logger = logging.getLogger(__name__)

class Select(DataFrameModule):
    def __init__(self, select_column='select', **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True),
                         SlotDescriptor('select', type=pd.DataFrame, required=False)])
        super(Select, self).__init__(**kwds)
        self.default_step_size = 1000
        self._select_column = select_column
        
    def run_step(self,run_number,step_size,howlong):
        select_slot = self.get_input_slot('select')
        df_slot = self.get_input_slot('df')
        if not select_slot:
            select = None
        else:
            select_df = select_slot.data()
            select_slot.update(run_number)
            if  select_slot.has_created(): # ignore deleted and updated
                df_slot.reset() # re-filter
            indices = select_slot.next_created() # read it all
            select = self.last_row(select_df)[self._select_column] # get the select expression
            if select is not None:
                if len(select)==0:
                    select=None
                else:
                    select = unicode(select) # make sure we have a string

        df_slot.update(run_number)
        if df_slot.has_deleted() or df_slot.has_updated():
            df_slot.reset()
            self._df = None
            df_slot.update(run_number)
        
        indices = df_slot.next_created(step_size)
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=steps)

        if select is None: # nothing to select, just pass through
            logger.info('No select, passing data through')
            self._df = df_slot.data()
            return self._return_run_step(self.state_blocked, steps_run=1)
        
        if isinstance(indices, slice):
            indices = slice(indices.start, indices.stop-1)
        df = df_slot.data().loc[indices]
        try:
            selected_df = df.eval(select)
            if isinstance(selected_df, pd.Series):
                selected_df = df[selected_df]
        except Exception as e:
            logger.error('Probably a syntax error in select expression: %s', e)
            self._df = df_slot.data()
            return self._return_run_step(self.state_blocked, steps_run=steps)
        selected_df[self.UPDATE_COLUMN] = run_number
        if self._df is None:
            self._df = selected_df
        else:
            self._df = self._df.append(selected_df) # don't ignore index I think
        return self._return_run_step(self.state_blocked, steps_run=steps)

    @staticmethod
    def make_range_query(column, low, high=None):
        if not is_valid_identifier(column):
            raise ProgressiveError('Cannot use column "%s", invalid name in expression',column)
        if high==None or low==high:
            return "({} == {})".format(low,column)
        elif low > high:
            low,high = high, low
        return "({} <= {} <= {})".format(low,column,high)

    @staticmethod
    def make_and_query(*expr):
        if len(expr)==1:
            return expr[0]
        elif len(expr)>1:
            return " and ".join(expr)
        return ""

    @staticmethod
    def make_or_query(*expr):
        if len(expr)==1:
            return expr[0]
        elif len(expr)>1:
            return " or ".join(expr)
        return ""
