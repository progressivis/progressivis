from progressive.core.common import ProgressiveError
from progressive.core.utils import typed_dataframe
from progressive.core.dataframe import DataFrameModule
from progressive.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

class Histogram2d(DataFrameModule):
    parameters = [('xbins',  np.dtype(int),   1024),
                  ('ybins',  np.dtype(int),   1024),
                  ('xmin',   np.dtype(float), 0),
                  ('xmax',   np.dtype(float), 1),
                  ('ymin',   np.dtype(float), 0),
                  ('ymax',   np.dtype(float), 1),
                  ('xdelta', np.dtype(float), 0),
                  ('ydelta', np.dtype(float), 0)]
                 
    def __init__(self, x_column, y_column, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame)])
        super(Histogram2d, self).__init__(dataframe_slot='histogram2d', **kwds)
        self.x_column = x_column
        self.y_column = y_column
        self.default_step_size = 10000

        columns = ['sum', 'histogram2d'] + [self.UPDATE_COLUMN]
        dtypes = [np.dtype(float), np.dtype(object)]
        values = [0, None, np.nan]

        self._df = typed_dataframe(columns, dtypes, values)

    def is_ready(self):
        if not self.get_input_slot('df').is_buffer_empty():
            return True
        return super(Histogram2d, self).is_ready()

    def run_step(self, step_size, howlong):
        dfslot = self.get_input_slot('df')
        input_df = dfslot.data()
        dfslot.update(self._start_time, input_df)
        if len(dfslot.deleted) or len(dfslot.updated) > len(dfslot.created):
            raise ProgressiveError('%s module does not manage updates or deletes', self.__class__.name)
        dfslot.buffer_created()

        indices = dfslot.next_buffered(step_size)
        steps = len(indices)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=steps)
        x = input_df.loc[indices, self.x_column]
        y = input_df.loc[indices, self.y_column]
        p = self.params
        histo, xedges, yedges = np.histogram2d(y, x,
                                               bins=[p.xbins, p.ybins],
                                               range=[[p.xmin, p.xmax],[p.ymin, p.ymax]],
                                               normed=False)
        sum = histo.sum()
        df = self._df
        old_histo = df.at[0, 'histogram2d']
        if old_histo is None:
            df.at[0, 'histogram2d'] = histo
            df.at[0, 'sum'] = sum
        else:
            old_histo += histo
            df.at[0, 'sum'] += sum
        df.at[0, self.UPDATE_COLUMN] = np.nan  # to update time stamps
        next_state = self.state_blocked if dfslot.is_buffer_empty() else self.state_ready
        print "Next state is %s" % next_state
        return self._return_run_step(next_state,
                                     steps_run=steps,
                                     reads=steps,
                                     updates=len(self._df))
