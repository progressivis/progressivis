from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd
import scipy as sp

from PIL import Image

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

    def predict_step_size(self, duration):
        # Module sample is constant time (supposedly)
        return 1

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
        image.transpose(Image.FLIP_TOP_BOTTOM)
        filename = p.filename
        if filename is not None:
            try:
                if re.search(r'%(0[\d])?d', filename):
                    filename = filename % (run_number)
                filename = self.storage.fullname(self, filename)
                image.save(filename)
                image = None
            except:
                raise

        values = [image, filename, run_number]
        df = self._df
        df.loc[run_number] = values
        if len(df) > p.history:
            self._df = df.loc[df.index[-p.history:]]
        return self._return_run_step(self.state_blocked,
                                     steps_run=1,
                                     reads=1,
                                     updates=1)

    def is_visualization(self):
        return True

    def get_visualization(self):
        return "heatmap";

    def to_json(self, short=False):
        json = super(Heatmap, self).to_json(short)
        if short:
            return json
        return self.heatmap_to_json(json, short)

    def heatmap_to_json(self, json, short):
        dfslot = self.get_input_slot('array')
        histo = dfslot.output_module
        json['columns'] = [histo.x_column, histo.y_column]
        histo_df = dfslot.data()
        if histo_df is not None and histo_df.index[-1] is not None:
            idx = histo_df.index[-1]
            row = histo_df.loc[idx]
            if not (np.isnan(row.xmin) or np.isnan(row.xmax)
                    or np.isnan(row.ymin) or np.isnan(row.ymax)):
                json['bounds'] = {
                    'xmin': row.xmin,
                    'ymin': row.ymin,
                    'xmax': row.xmax,
                    'ymax': row.ymax
                }
        df = self._df
        if df is not None and self._last_update is not None:
            json['image'] = "%s/image?run_number=%d"%(self.id,self._last_update)
        return json

    def get_image(self, run_number=None):
        if self._df is None:
            return None
        if run_number is None:
            run_number = self._df.index[-1]
        if run_number is None:
            return None

        if run_number in self._df.index:
            (image, filename, run_number) = self._df.loc[run_number]
            return filename if filename is not None else image
        else:
            return None
