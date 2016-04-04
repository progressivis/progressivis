from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor
from progressivis.core.utils import last_row, create_dataframe

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

class RangeQuery(DataFrameModule):
    schema = [('query', np.dtype(object), None),
              DataFrameModule.UPDATE_COLUMN_DESC]

    def __init__(self, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('min', type=pd.DataFrame, required=True),
                         SlotDescriptor('max', type=pd.DataFrame, required=True),
                         SlotDescriptor('min_value', type=pd.DataFrame, required=True),
                         SlotDescriptor('max_value', type=pd.DataFrame, required=True)])
        self._add_slots(kwds, 'output_descriptors',
                        [SlotDescriptor('min', type=pd.DataFrame, required=False),
                         SlotDescriptor('max', type=pd.DataFrame, required=False)])
        super(RangeQuery, self).__init__(dataframe_slot='query', **kwds)
        self.default_step_size = 1
        self._df = create_dataframe(RangeQuery.schema, empty=True)
        self._min = None
        self._max = None

    def create_dependent_modules(self,
                                 input_module,
                                 input_slot,
                                 min_value=None,
                                 max_value=None,
                                 **kwds):
        from progressivis.io import Variable
        from progressivis.stats import Min, Max

        if hasattr(self, 'input_module'): # test if already called
            return self

        s = self.scheduler()
        self.input_module = input_module
        self.input_slot = input_slot

        min = Min(group=self.id, scheduler=s)
        min.input.df = input_module.output[input_slot]
        max = Max(group=self.id, scheduler=s)
        max.input.df = input_module.output[input_slot]
        if min_value is None:
            min_value = Variable(group=self.id, scheduler=s)
            min_value.input.like = min.output.df

        if max_value is None:
            max_value = Variable(group=self.id, scheduler=s)
            max_value.input.like = max.output.df

        range_query = self
        range_query.input.min = min.output.df
        range_query.input.max = max.output.df
        range_query.input.min_value = min_value.output.df # might fail if min_value is not a Min
        range_query.input.max_value = max_value.output.df # might fail if max_value is not a Max

        self.min = min
        self.max = max
        self.min_value = min_value
        self.max_value = max_value
        return range_query

    def is_visualization(self):
        return True

    def get_visualization(self):
        return "range_query"

    def to_json(self, short=False):
        json = super(RangeQuery, self).to_json(short)
        if short:
            return json
        return self._ranges_to_json(json)

    def _ranges_to_json(self, json):
        #join the min and max input slots, and the min and max output slots by name
        #example:
        #ranges = [{"name": "xRange", "in_min": 0, "in_max": 1, "out_min": 0, "out_max": 1},
        #    {"name": "yRange", "in_min": 0, "in_max": 1, "out_min": 0, "out_max": 1}]
        in_min = self.get_input_slot('min').data()
        in_max = self.get_input_slot('max').data()
        out_min = self.get_data('min')
        out_max = self.get_data('max')
        if all(x is not None for x in [in_min, in_max, out_min, out_max]):
            in_min_final = last_row(in_min, remove_update=True)
            in_max_final = last_row(in_max, remove_update=True)
            out_min_final = last_row(out_min, remove_update=True)
            out_max_final = last_row(out_max, remove_update=True)
            ranges = pd.DataFrame({'in_min': in_min_final,
                                   'in_max': in_max_final,
                                   'out_min': out_min_final,
                                   'out_max': out_max_final})
            ranges.index.name = "name"
            json['ranges'] = ranges.reset_index().to_dict(orient='records')
        return json

    def get_data(self, name):
        if name == 'min':
            return self._min
        elif name == 'max':
            return self._max
        return super(RangeQuery, self).get_data(name)

    def run_step(self, run_number, step_size, howlong):
        # Assuming min and max come from applying Min and Max to a DataFrame with e.g.
        # columns 'a' and 'b', we now have min containing the 2 columns and max too.
        # min_value and max_value are generated from an interaction probably, so
        # they contain either no value (e.g. index only contains 'a', 'b', or empty),
        # or NaN for 'a' and/or 'b'.
        min_slot = self.get_input_slot('min')
        with min_slot.lock:
            min_slot.update(run_number)
            min = last_row(min_slot.data(), remove_update=True)
        max_slot = self.get_input_slot('max')
        with max_slot.lock:
            max_slot.update(run_number)
            max = last_row(max_slot.data(), remove_update=True)
        minv_slot = self.get_input_slot('min_value')
        with minv_slot.lock:
            minv_slot.update(run_number)
            minv = last_row(minv_slot.data(), remove_update=True)
        if minv is None:
            minv = min
        maxv_slot = self.get_input_slot('max_value')
        with maxv_slot.lock:
            maxv_slot.update(run_number)
            maxv = last_row(maxv_slot.data(), remove_update=True)
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
        op = aligned.loc[:, ['min', 'min_value']].max(axis=1)
        op[self.UPDATE_COLUMN] = run_number
        op.name = 'min'
        self._min = pd.DataFrame([op], index=[run_number])

        op = aligned.loc[:, ['max', 'max_value']].min(axis=1)
        op[self.UPDATE_COLUMN] = run_number
        op.name = 'max'
        self._max = pd.DataFrame([op], index=[run_number])

        with self.lock:
            if len(self._df) != 0:
                last = self._df.at[self._df.index[-1], 'query']
                if last == query: # do not repeat the query to allow optimizing downstream
                    return self._return_run_step(self.state_blocked, steps_run=1)
                logger.info('New query: "%s"', query)
            self._df.loc[run_number] = pd.Series({'query': query, self.UPDATE_COLUMN: run_number})
        return self._return_run_step(self.state_blocked, steps_run=1)
