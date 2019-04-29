"""
Visualize a data table with histograms.
"""
from __future__ import print_function

import logging

import numbers
import six
import numpy as np

from progressivis.table.nary import NAry
from progressivis.core.slot import SlotDescriptor
from progressivis.stats import Histogram1D
from progressivis.table.table import BaseTable

logger = logging.getLogger(__name__)


class Histograms(NAry):
    "Visualize a table with multiple histograms"
    parameters = [('bins', np.dtype(int), 128),
                  ('delta', np.dtype(float), -5)] # 5%

    def __init__(self, columns=None, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('min', type=BaseTable, required=True),
                         SlotDescriptor('max', type=BaseTable, required=True)])
        self._add_slots(kwds, 'output_descriptors',
                        [SlotDescriptor('min', type=BaseTable, required=False),
                         SlotDescriptor('max', type=BaseTable, required=False)])
        super(Histograms, self).__init__(**kwds)
        self.default_step_size = 1
        self._columns = columns
        self._histogram = {}

    def table(self):
        "Return the table"
        return self.get_input_slot('table').data()

    def get_data(self, name):
        if name == 'min':
            return self.get_input_slot('min').data()
        if name == 'max':
            return self.get_input_slot('max').data()
        return super(Histograms, self).get_data(name)

    def predict_step_size(self, duration):
        return 1

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('table')
        input_df = dfslot.data()
        dfslot.update(run_number)
        dfslot.clear_buffers()
        col_changes = dfslot.column_changes
        if col_changes is not None:
            self._create_columns(col_changes.created, input_df)
            self._delete_columns(col_changes.deleted)
        return self._return_run_step(self.state_blocked, steps_run=1)

    def _create_columns(self, columns, df):
        bins = self.params.bins
        delta = self.params.delta # crude
        inp = self.get_input_module('table')
        minmod = self.get_input_module('min')
        maxmod = self.get_input_module('max')
        for column in columns:
            #if c==UPDATE_COLUMN:
            #    continue
            logger.debug('Creating histogram1d %s', column)
            dtype = df[column].dtype
            if not np.issubdtype(dtype, numbers.Number):
                # only create histograms for number columns
                continue
            histo = Histogram1D(group=self.name, column=column,
                                bins=bins, delta=delta, dataflow=self.daataflow)
            histo.input.table = inp.output.table
            histo.input.min = minmod.output.table
            histo.input.max = maxmod.output.table
            self.input.table = histo.output._trace # will become table.1 ...
            self._histogram[column] = histo

    def _delete_columns(self, columns):
        for column in columns:
            logger.debug('Deleting histogram1d %s', column)
            histo = self._histogram[column]
            del self._histogram[column]
            histo.destroy()

    def is_visualization(self):
        return True

    def get_visualization(self):
        return "histograms"

    def to_json(self, short=False):
        json = super(Histograms, self).to_json(short)
        if short:
            return json
        return self._histograms_to_json(json)

    def _histograms_to_json(self, json):
        histo_json = {}
        for (column, value) in six.iteritems(self._histogram):
            column = six.u(column)
            histo_json[column] = value.get_histogram()
        json['histograms'] = histo_json
        return json
