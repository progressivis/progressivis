"""
Visualization of a Histogram2D as a heaatmap
"""
import re
import logging
import base64
import io
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
    inputs = [SlotDescriptor('array', type=Table)]

    # schema = [('image', np.dtype(object), None),
    #           ('filename', np.dtype(object), None),
    #           UPDATE_COLUMN_DESC]
    schema = "{filename: string, time: int64}"

    def __init__(self, colormap=None, **kwds):
        super(Heatmap, self).__init__(**kwds)
        self.colormap = colormap
        self.default_step_size = 1

        name = self.generate_table_name('Heatmap')
        # params = self.params
        # if params.filename is None:
        #     params.filename = name+'%d.png'
        self.result = Table(name, dshape=Heatmap.schema, create=True)

    def predict_step_size(self, duration):
        _ = duration
        # Module sample is constant time (supposedly)
        return 1

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('array')
        input_df = dfslot.data()
        # dfslot.update(run_number)
        dfslot.deleted.next()
        indices = dfslot.created.next()
        steps = indices_len(indices)
        if steps == 0:
            indices = dfslot.updated.next()
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=1)
        histo = input_df.last()['array']
        if histo is None:
            return self._return_run_step(self.state_blocked, steps_run=1)
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
            #import pdb;pdb.set_trace()
            if cmin is None:
                cmin = histo.min()
            if cmax is None:
                cmax = histo.max()
            cscale = cmax - cmin
            scale = float(high - low) / cscale
            data = (sp.special.cbrt(histo) * 1.0 - cmin) * scale + 0.4999
            data[data > high] = high
            data[data < 0] = 0
            data = np.cast[np.uint32](data)
            if low != 0:
                data += low

            image = Image.fromarray(data, mode='I')
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            filename = params.filename
        except:
            image = None
            filename = None
        if filename is not None:
            try:
                if re.search(r'%(0[\d])?d', filename):
                    filename = filename % (run_number)
                filename = self.storage.fullname(self, filename)
                # TODO should do it atomically since it will be
                # called 4 times with the same fn
                image.save(filename, format='PNG')  # bits=16)
                logger.debug('Saved image %s', filename)
                image = None
            except:
                logger.error('Cannot save image %s', filename)
                raise
        else:
            buffered = io.BytesIO()
            image.save(buffered, format='PNG', bits=16)
            res = str(base64.b64encode(buffered.getvalue()), "ascii")
            filename = "data:image/png;base64,"+res

        if len(self.result) == 0 or self.result.last()['time'] != run_number:
            values = {'filename': filename, 'time': run_number}
            self.result.add(values)
        return self._return_run_step(self.state_blocked, steps_run=1)

    def is_visualization(self):
        return False

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
        df = self.result
        if df is not None and self._last_update != 0:
            row = df.last()
            json['image'] = row['filename']
        return json

    def get_image(self, run_number=None):
        if self.result is None or len(self.result) == 0:
            return None
        last = self.result.last()
        if run_number is None or run_number >= last['time']:
            run_number = last['time']
            filename = last['filename']
        else:
            time = self.result['time']
            idx = np.where(time == run_number)[0]
            if len(idx) == 0:
                filename = last['filename']
            else:
                filename = self.result['filename'][idx[0]]

        return filename
