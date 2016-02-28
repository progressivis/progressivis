from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor
from progressivis.stats import Histogram1D

import pandas as pd
import numpy as np
import numbers

import logging
logger = logging.getLogger(__name__)


class Histograms(DataFrameModule):
    def __init__(self, columns=None, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True),
                         SlotDescriptor('min', type=pd.DataFrame, required=True),
                         SlotDescriptor('max', type=pd.DataFrame, required=True)])
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor('min', type=pd.DataFrame, required=False),
                         SlotDescriptor('max', type=pd.DataFrame, required=False)])
        super(Histograms, self).__init__(dataframe_slot='df', **kwds)
        self.default_step_size = 1
        self._columns = columns
        self._histogram = {}

    def df(self):
        return self.get_input_slot('df').data()

    def get_data(self, name):
        if name=='min':
            return self.get_input_slot('min').data()
        if name=='max':
            return self.get_input_slot('max').data()
        return super(Histograms, self).get_data(name)

    def predict_step_size(self, duration):
        return 1

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        input_df = dfslot.data()
        dfslot.update(run_number)
        col_changes = dfslot.column_changes
        if col_changes is not None:
            self.create_columns(col_changes.created, dfslot.data())
            self.delete_columns(col_changes.deleted)
        return self._return_run_step(self.state_blocked, steps_run=1)

    def create_columns(self, columns, df):
        for c in columns:
            if c==self.UPDATE_COLUMN:
                continue
            print 'Creating histogram1d ',c
            dtype = df[c].dtype
            if not np.issubdtype(dtype, numbers.Number):
                # only create histograms for number columns
                continue
            h = Histogram1D(group=self.id, column=c,scheduler=self.scheduler())
            h.input.df = self.output.df
            h.input.min = self.output.min
            h.input.max = self.output.max
            self._histogram[c] = h

    def delete_columns(self, columns):
        for c in columns:
            print 'Deleting histogram1d ',c
            h = self._histogram[c]
            del self._histogram[c]
            h.destroy()

    def is_visualization(self):
        return True

    def get_visualization(self):
        return "histograms";

    def to_json(self, short=False):
        json = super(Histograms, self).to_json(short)
        if short:
            return json
        return self.histograms_to_json(json, short)

    def histograms_to_json(self, json, short):
        # dfslot = self.get_input_slot('array')
        # histo = dfslot.output_module
        # json['columns'] = [histo.x_column, histo.y_column]
        # with dfslot.lock:
        #     histo_df = dfslot.data()
        #     if histo_df is not None and histo_df.index[-1] is not None:
        #         row = self.last_row(histo_df)
        #         if not (np.isnan(row.xmin) or np.isnan(row.xmax)
        #                 or np.isnan(row.ymin) or np.isnan(row.ymax)):
        #             json['bounds'] = {
        #                 'xmin': row.xmin,
        #                 'ymin': row.ymin,
        #                 'xmax': row.xmax,
        #                 'ymax': row.ymax
        #             }
        # with self.lock:
        #     df = self.df()
        #     if df is not None and self._last_update is not None:
        #         row = self.last_row(df)
        #         json['image'] = "/progressivis/module/image/%s?run_number=%d"%(self.id,row[self.UPDATE_COLUMN])
        return json

