from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

class RangeQuery(DataFrameModule):
    schema = [('query', np.dtype(object), None),
              DataFrameModule.UPDATE_COLUMN_DESC]

    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('min', type=pd.DataFrame, required=True),
                         SlotDescriptor('max', type=pd.DataFrame, required=True),
                         SlotDescriptor('min_value', type=pd.DataFrame, required=True),
                         SlotDescriptor('max_value', type=pd.DataFrame, required=True)])
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor('min', type=pd.DataFrame, required=False),
                         SlotDescriptor('max', type=pd.DataFrame, required=False)])
        super(RangeQuery, self).__init__(dataframe_slot='query', **kwds)
        self.default_step_size = 1
        self._df = self.create_dataframe(RangeQuery.schema, empty=True)
        self._min = None
        self._max = None

    def is_visualization(self):
        return True

    def get_visualization(self):
        return "range_query"

    def get_data(self, name):
        if name=='min':
            return self._min
        elif name=='max':
            return self._max
        return super(RangeQuery,self).get_data(name)

    def run_step(self,run_number,step_size, howlong):
        # Assuming min and max come from applying Min and Max to a DataFrame with e.g.
        # columns 'a' and 'b', we now have min containing the 2 columns and max too.
        # min_value and max_value are generated from an interaction probably, so
        # they contain either no value (e.g. index only contains 'a', 'b', or empty),
        # or NaN for 'a' and/or 'b'.
        min_slot = self.get_input_slot('min')
        min_slot.update(run_number)
        min = self.last_row(min_slot.data(), remove_update=True)
        max_slot = self.get_input_slot('max')
        max_slot.update(run_number)
        max = self.last_row(max_slot.data(), remove_update=True)
        minv_slot = self.get_input_slot('min_value')
        minv_slot.update(run_number)
        minv = self.last_row(minv_slot.data(), remove_update=True)
        if minv is None:
            mminv = min
        maxv_slot = self.get_input_slot('max_value')
        maxv_slot.update(run_number)
        maxv = self.last_row(maxv_slot.data(), remove_update=True)
        if maxv is None:
            maxv = max

        # Need to align the series to create queries
        aligned = pd.DataFrame({'min': min, 'max': max, 'min_value': minv, 'max_value': maxv})
        min_query = aligned['min_value'] > aligned['min']
        max_query = aligned['max_value'] < aligned['max']
        range_query = min_query & max_query
        min_query = min_query & (~ range_query)
        max_query = max_query & (~ range_query)
        query = ''
        for row in aligned.index[min_query]:
            if query: query += ' and '
            query += '({} < {})'.format(minv[row], row)
        for row in aligned.index[max_query]:
            if query: query += ' and '
            query += '({} < {})'.format(row, maxv[row])
        for row in aligned.index[range_query]:
            if query: query += ' and '
            query += '({} < {} < {})'.format(minv[row], row, maxv[row])

        # compute the new min/max columns
        op = aligned[['min','min_value']].max(axis=1)
        op[self.UPDATE_COLUMN] = run_number
        op.name = 'min'
        self._min = pd.DataFrame([op],index=[run_number])

        op = aligned[['max','max_value']].min(axis=1)
        op[self.UPDATE_COLUMN] = run_number
        op.name = 'max'
        self._max = pd.DataFrame([op],index=[run_number])

        if len(self._df)!=0:
            last = self._df.at[self._df.index[-1],'query']
            if last==query: # do not repeat the query to allow optimizing downstream
                return self._return_run_step(self.state_blocked, steps_run=1)
        self._df.loc[run_number] = pd.Series({'query': query, self.UPDATE_COLUMN: run_number})
        return self._return_run_step(self.state_blocked, steps_run=1)
