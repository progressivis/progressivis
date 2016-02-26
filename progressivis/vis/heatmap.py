from progressivis.core.utils import indices_len
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd
import scipy as sp

from PIL import Image

import re

import logging
logger = logging.getLogger(__name__)


class Heatmap(DataFrameModule):
    parameters = [('cmax', np.dtype(float), np.nan),
                  ('cmin', np.dtype(float), np.nan),
                  ('high', np.dtype(int),   65536),
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
        dfslot.update(run_number)
        indices = dfslot.next_created()
        steps = indices_len(indices)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=1)
        with dfslot.lock:
            histo = input_df.at[input_df.index[-1], 'array']
        if histo is None:
            return self._return_run_step(self.state_blocked, steps_run=1)
        p = self.params
        cmax = p.cmax
        if np.isnan(cmax):
            cmax = None
        cmin = p.cmin
        if np.isnan(cmin):
            cmin = None
        high = p.high
        low = p.low
        try:
            image = sp.misc.toimage(sp.special.cbrt(histo), cmin=cmin, cmax=cmax, high=high, low=low, mode='I')
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            filename = p.filename
        except:
            image = None
            filename = None
        if filename is not None:
            try:
                if re.search(r'%(0[\d])?d', filename):
                    filename = filename % (run_number)
                filename = self.storage.fullname(self, filename)
                image.save(filename, format='PNG', bits=16)
                logger.debug('Saved image %s', filename)
                image = None
            except:
                logger.error('Cannot save image %s', filename)
                raise

        values = [image, filename, run_number]
        with self.lock:
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
        with dfslot.lock:
            histo_df = dfslot.data()
            if histo_df is not None and histo_df.index[-1] is not None:
                row = self.last_row(histo_df)
                if not (np.isnan(row.xmin) or np.isnan(row.xmax)
                        or np.isnan(row.ymin) or np.isnan(row.ymax)):
                    json['bounds'] = {
                        'xmin': row.xmin,
                        'ymin': row.ymin,
                        'xmax': row.xmax,
                        'ymax': row.ymax
                    }
        with self.lock:
            df = self.df()
            if df is not None and self._last_update is not None:
                row = self.last_row(df)
                json['image'] = "/progressivis/module/image/%s?run_number=%d"%(self.id,row[self.UPDATE_COLUMN])
        return json

    def get_image(self, run_number=None):
        if self._df is None or len(self._df)==0:
            return None
        last_run_number = self._df.index[-1]
        if run_number is None or run_number > last_run_number:
            run_number = last_run_number

        rec = self._df.loc[run_number]
        image = rec['image']
        filename = rec['filename']
        return filename if filename is not None else image
