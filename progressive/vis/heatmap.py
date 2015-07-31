from progressive.core.common import ProgressiveError
from progressive.core.dataframe import DataFrameModule
from progressive.core.slot import SlotDescriptor

import numpy as np
import pandas as pd
import scipy as sp

import re

class Heatmap(DataFrameModule):
    parameters = [('cmax', np.dtype(float), np.nan),
                  ('cmin', np.dtype(float), np.nan),
                  ('high', np.dtype(int),   255),
                  ('low',  np.dtype(int),   0),
                  ('filename', np.dtype(object), None),
                  ('history', np.dtype(int), 3) ]

    schema = [('image', np.dtype(object), None),
              ('filename', np.dtype(object), None),
              DataFrameModule.UPDATE_COLUMN_DESC]
                 
    def __init__(self, colormap=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('array', type=pd.DataFrame)])
        super(Heatmap, self).__init__(dataframe_slot='heatmap', **kwds)
        self.colormap = colormap
        self.default_step_size = 1

        self._df = self.create_dataframe(Heatmap.schema)

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('array')
        input_df = dfslot.data()
        dfslot.update(run_number, input_df)
        histo = input_df.at[input_df.index[-1], 'array']
        p = self.params
        cmax = p.cmax
        if np.isnan(cmax):
            cmax = None
        cmin = p.cmin
        if np.isnan(cmin):
            cmin = None
        high = p.high
        low = p.low
        image = sp.misc.toimage(histo, cmin=cmin, cmax=cmax, high=high, low=low)
        filename = p.filename
        if filename is not None:
            try:
                if re.search(r'%(0[\d])?d', filename):
                    filename = filename % (run_number)
                image.save(filename)
            except:
                pass # discard error for now
        values = [image, filename, run_number]
        df = self._df
        df.loc[run_number] = values
        if len(df) > p.history:
            self._df = df.loc[df.index[-p.history:]]
        return self._return_run_step(self.state_blocked,
                                     steps_run=1,
                                     reads=1,
                                     updates=1)
