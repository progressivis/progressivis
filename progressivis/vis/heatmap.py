"""
Visualization of a Histogram2D as a heaatmap
"""
import re
import logging
import base64

import six
import numpy as np
import scipy as sp
from PIL import Image

from progressivis.core.utils import indices_len
from progressivis.core.slot import SlotDescriptor
from progressivis.table import Table
from progressivis.table.module import TableModule


logger = logging.getLogger(__name__)


class Heatmap(TableModule):
    "Heatmap module"
    parameters = [('cmax', np.dtype(float), np.nan),
                  ('cmin', np.dtype(float), np.nan),
                  ('high', np.dtype(int), 65536),
                  ('low', np.dtype(int), 0),
                  ('filename', np.dtype(object), None),
                  ('history', np.dtype(int), 3)]

    # schema = [('image', np.dtype(object), None),
    #           ('filename', np.dtype(object), None),
    #           UPDATE_COLUMN_DESC]
    schema = "{filename: string, time: int64}"

    def __init__(self, colormap=None, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('array', type=Table)])
        super(Heatmap, self).__init__(table_slot='heatmap', **kwds)
        self.colormap = colormap
        self.default_step_size = 1
        self._img_cache = None
        name = self.generate_table_name('Heatmap')
        params = self.params
        # if params.filename is None:
        #     params.filename = name+'%d.png'
        self._table = Table(name, dshape=Heatmap.schema, create=True)

    def predict_step_size(self, duration):
        _ = duration
        # Module sample is constant time (supposedly)
        return 1

    def run_step(self, run_number, step_size, howlong):
        s = self.scheduler()
        #if s.has_input() and not s._shortcut_once_again:
        #    return self._return_run_step(self.state_blocked, steps_run=1)
        dfslot = self.get_input_slot('array')
        input_df = dfslot.data()
        dfslot.update(run_number)
        indices = dfslot.created.next()
        steps = indices_len(indices)
        if steps == 0:
            indices = dfslot.updated.next()
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=1)
        self._img_cache = None
        return self._return_run_step(self.state_blocked, steps_run=1)


    def process_img(self):
        #if self.scheduler().has_input():
        #    import pdb;pdb.set_trace()
        if self._img_cache:
            return self._img_cache
        dfslot = self.get_input_slot('array')
        input_df = dfslot.data()
        with dfslot.lock:
            last_row = input_df.last()
            if not last_row:
                return None
            histo = last_row['array']
        if histo is None:
            return None
        params = self.params
        cmax = params.cmax
        if np.isnan(cmax):
            cmax = None
        cmin = params.cmin
        if np.isnan(cmin):
            cmin = None
        high = params.high
        low = params.low
        try:
            #print("heatmap works ********************", self.scheduler().has_input())
            image = sp.misc.toimage(sp.special.cbrt(histo),
                                    cmin=cmin, cmax=cmax,
                                    high=high, low=low, mode='I')
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        except:
            return None
        buffered = six.BytesIO()
        image.save(buffered, format='PNG', bits=16)
        res = base64.b64encode(buffered.getvalue())
        if six.PY3:
            res = str(base64.b64encode(buffered.getvalue()), "ascii")
        self._img_cache = "data:image/png;base64,"+res
        return self._img_cache
    
    def is_visualization(self):
        return True

    def get_visualization(self):
        return "heatmap"

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
            if histo_df is not None and len(histo_df) != 0:
                row = histo_df.last()
                if not (np.isnan(row['xmin']) or np.isnan(row['xmax'])
                        or np.isnan(row['ymin']) or np.isnan(row['ymax'])):
                    json['bounds'] = {
                        'xmin': row['xmin'],
                        'ymin': row['ymin'],
                        'xmax': row['xmax'],
                        'ymax': row['ymax']
                    }
        json['image'] = self.process_img()
        return json

