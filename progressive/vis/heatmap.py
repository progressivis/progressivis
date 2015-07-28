from progressive.core.common import ProgressiveError
from progressive.core.utils import typed_dataframe
from progressive.core.dataframe import DataFrameModule
from progressive.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

class Heatmap(DataFrameModule):
    parameters = [('cmax', np.dtype(float), np.nan),
                  ('cmin', np.dtype(float), np.nan),
                  ('high', np.dtype(int),   255),
                  ('low',  np.dtype(int),   0),
                  ('filename', np.dtype(object), None)]
                 
    def __init__(self, colormap=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame)])
        super(Heatmap, self).__init__(dataframe_slot='heatmap', **kwds)
        self.colormap = colormap
        self.default_step_size = 1

        columns = ['image'] + [self.UPDATE_COLUMN]
        dtypes = [np.dtype(object), np.dtype(float)]
        values = [None, np.nan]

        self._df = typed_dataframe(columns, dtypes, values)

    def is_ready(self):
        if not self.get_input_slot('df').is_buffer_empty():
            return True
        return super(Heatmap, self).is_ready()

    def run_step(self, step_size, howlong):
        dfslot = self.get_input_slot('df')
        input_df = dfslot.data()
        dfslot.update(self._start_time, input_df)
        histo = input_df.at[0, 'histogram2d']
        p = self.params
        cmax = p.cmax
        if cmax == np.nan:
            cmax = None
        cmin = p.cmin
        if cmin == np.nan:
            cmin = None
        high = p.high
        low = p.low
        image = histo.toimage(histo, cmin=cmin, cmax=cmax, high=high, low=low)
        df = self._df
        df.at[0, 'image'] = image
        df.at[0, self.UPDATE_COLUMN] = np.nan  # to update time stamps
        return self._return_run_step(self.state_blocked,
                                     steps_run=1,
                                     reads=1,
                                     updates=1)
