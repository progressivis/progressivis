from progressivis.core.common import ProgressiveError
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

class Histogram2D(DataFrameModule):
    parameters = [('xbins',  np.dtype(int),   512),
                  ('ybins',  np.dtype(int),   512),
                  ('xmin',   np.dtype(float), 0),
                  ('xmax',   np.dtype(float), 1),
                  ('ymin',   np.dtype(float), 0),
                  ('ymax',   np.dtype(float), 1),
                  ('xdelta', np.dtype(float), -5), # means 5%
                  ('ydelta', np.dtype(float), -5), # means 5%
                  ('history',np.dtype(int),   3) ]

    schema = [('array', np.dtype(object), None),
              ('cmin', np.dtype(float), np.nan),
              ('cmax', np.dtype(float), np.nan),
              ('xmin',   np.dtype(float), np.nan),
              ('xmax',   np.dtype(float), np.nan),
              ('ymin',   np.dtype(float), np.nan),
              ('ymax',   np.dtype(float), np.nan),
              DataFrameModule.UPDATE_COLUMN_DESC]

    def __init__(self, x_column, y_column, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame)])
        super(Histogram2D, self).__init__(dataframe_slot='histogram2d', **kwds)
        self.x_column = x_column
        self.y_column = y_column
        self.default_step_size = 10000
        self.total_read = 0
        self._old_histo = None
        self._bounds = None
        self._df = self.create_dataframe(Histogram2D.schema)

    def update_bounds(self):
        p = self.params
        logger.info('Updating bounds')
        xdelta = p.xdelta
        ydelta = p.ydelta
        xmin = p.xmin
        xmax = p.xmax
        ymin = p.ymin
        ymax = p.ymax
        if xmax < xmin:
            xmax, xmin = xmin, xmax
            logger.warn('xmax < xmin, swapped')
        if ymax < ymin:
            ymax, ymin = ymin, ymax
            logger.warn('ymax < ymin, swapped')
        if xdelta < 0:
            dx = xmax - xmin
            xdelta = dx*xdelta/-100.0
            logger.info('xdelta is %f', xdelta)
        if ydelta < 0:
            dy = p.ymax - p.ymin
            ydelta = dy*ydelta/-100.0
            logger.info('ydelta is %f', ydelta)
        return (xmin-xdelta, xmax+xdelta, ymin-ydelta, ymax+ydelta)

    def is_ready(self):
        if not self.get_input_slot('df').is_buffer_empty():
            return True
        return super(Histogram2D, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        input_df = dfslot.data()
        df = self._df
        old_histo = self._old_histo
        p = self.params # TODO: check if params have changed
        bounds_changed = False
        if self._bounds is None:
            self._bounds = self.update_bounds()
            xmin, xmax, ymin, ymax = self._bounds
        else:
            xmin, xmax, ymin, ymax = self._bounds
            # If new bounds extend new ones including deltas, invalidate
            if p.xmin < xmin or p.xmax > xmax or p.ymin < ymin or p.ymax > ymax:
                bounds_changed = True
                self._bounds = self.update_bounds()
                xmin, xmax, ymin, ymax = self._bounds
        
        dfslot.update(run_number, input_df)
        if bounds_changed or len(dfslot.deleted) or len(dfslot.updated) > len(dfslot.created):
            dfslot.reset()
            self.total_read = 0

        dfslot.buffer_created()

        indices = dfslot.next_buffered(step_size)
        steps = len(indices)
        if steps == 0:
            self._old_histo = old_histo = None # should store the old histo now
            logger.info('Index buffer empty')
            return self._return_run_step(self.state_blocked, steps_run=steps)

        self.total_read += steps
        x = input_df.loc[indices, self.x_column]
        y = input_df.loc[indices, self.y_column]
        histo, xedges, yedges = np.histogram2d(y, x,
                                               bins=[p.xbins, p.ybins],
                                               range=[[xmin, xmax],[ymin, ymax]],
                                               normed=False)
        if old_histo is None:
            old_histo = histo
        else:
            old_histo += histo
        cmax = old_histo.max()
        values = [old_histo, 0, cmax, xmin, xmax, ymin, ymax, run_number]
        df.loc[run_number] = values
        if len(df) > p.history:
            self._df = df.loc[df.index[-p.history:]]
        self._old_histo = old_histo
        return self._return_run_step(dfslot.next_state(),
                                     steps_run=steps, reads=steps, updates=len(self._df))
