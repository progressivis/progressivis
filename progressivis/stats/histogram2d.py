from progressivis.core.common import indices_len
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

class Histogram2D(DataFrameModule):
    parameters = [('xbins',  np.dtype(int),   512),
                  ('ybins',  np.dtype(int),   512),
#                  ('xmin',   np.dtype(float), 0),
#                  ('xmax',   np.dtype(float), 1),
#                  ('ymin',   np.dtype(float), 0),
#                  ('ymax',   np.dtype(float), 1),
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
                        [SlotDescriptor('df', type=pd.DataFrame, required=True),
                         SlotDescriptor('min', type=pd.DataFrame, required=True),
                         SlotDescriptor('max', type=pd.DataFrame, required=True)])
        super(Histogram2D, self).__init__(dataframe_slot='df', **kwds)
        self.x_column = x_column
        self.y_column = y_column
        self.default_step_size = 10000
        self.total_read = 0
        self._old_histo = None
        self._bounds = None
        self._df = self.create_dataframe(Histogram2D.schema)

    def get_bounds(self, run_number):
        min_slot = self.get_input_slot('min')
        min_slot.update(run_number)
        min_slot.next_created()
        min_df = min_slot.data()
        if len(min_df)==0 and self._bounds is None:
            return None
        min = min_df.loc[min_df.index[-1]]
        xmin = min[self.x_column]
        ymin = min[self.y_column]
        
        max_slot = self.get_input_slot('max')
        max_slot.update(run_number)
        max_slot.next_created()
        max_df = max_slot.data()
        if len(max_df)==0 and self._bounds is None:
            return None
        max = max_df.loc[max_df.index[-1]]
        xmax = max[self.x_column]
        ymax = max[self.y_column]
        
        if xmax < xmin:
            xmax, xmin = xmin, xmax
            logger.warn('xmax < xmin, swapped')
        if ymax < ymin:
            ymax, ymin = ymin, ymax
            logger.warn('ymax < ymin, swapped')
        return (xmin, xmax, ymin, ymax)

    def get_delta(self, xmin, xmax, ymin, ymax):
        p = self.params
        xdelta, ydelta = p[['xdelta', 'ydelta']]
        if xdelta < 0:
            dx = xmax - xmin
            xdelta = dx*xdelta/-100.0
            logger.info('xdelta is %f', xdelta)
        if ydelta < 0:
            dy = ymax - ymin
            ydelta = dy*ydelta/-100.0
            logger.info('ydelta is %f', ydelta)
        return (xdelta, ydelta)

    def is_ready(self):
        # If we have created data but no valid min/max, we can only wait
        if self._bounds and self.get_input_slot('df').has_created():
            return True
        return super(Histogram2D, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        input_df = dfslot.data()
        dfslot.update(run_number)
        if dfslot.has_updated() or dfslot.has_deleted():
            dfslot.reset()
            dfslot.update(run_number)
            self.total_read = 0

        if not dfslot.has_created(): # nothing to do, just wait 
            logger.info('Index buffer empty')
            return self._return_run_step(self.state_blocked, steps_run=0)
        old_histo = self._old_histo
            
        bounds = self.get_bounds(run_number)
        if bounds is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        xmin, xmax, ymin, ymax = bounds
        if self._bounds is None:
            (xdelta, ydelta) = self.get_delta(*bounds)
            self._bounds = (xmin-xdelta,xmax+xdelta,ymin-ydelta,ymax+ydelta)
        else:
            (dxmin, dxmax, dymin, dymax) = self._bounds
            if xmin < dxmin or xmax > dxmax or ymin < dymin or ymax > dymax:
                (xdelta, ydelta) = self.get_delta(*bounds)
                self._bounds = (xmin-xdelta,xmax+xdelta,ymin-ydelta,ymax+ydelta)
                logger.info('Updated bounds: %s', self._bounds)
                dfslot.reset()
                dfslot.update(run_number) # should recompute the histogram from scatch
                old_histo = None 
        
        xmin, xmax, ymin, ymax = self._bounds

        # Now, we know we have data and bounds, proceed to create a new histogram

        indices = dfslot.next_created(step_size)
        steps = indices_len(indices)
        logger.info('Read %d rows', steps)
        self.total_read += steps
        
        if isinstance(indices,slice):
            filtered_df = input_df.loc[indices.start:indices.stop-1] # semantic of slice with .loc
        else:
            filtered_df = input_df.loc[indices]
        x = filtered_df[self.x_column]
        y = filtered_df[self.y_column]
        p = self.params
        if len(x)>0:
            histo, xedges, yedges = np.histogram2d(y, x,
                                                   bins=[p.xbins, p.ybins],
                                                   range=[[xmin, xmax],[ymin, ymax]],
                                                   normed=False)
        else:
            histo = np.array([p.xbins, p.ybins], dtype=int)
            cmax = 0

        if old_histo is None:
            old_histo = histo
        else:
            old_histo += histo

        cmax = old_histo.max()
        values = [old_histo, 0, cmax, xmin, xmax, ymin, ymax, run_number]
        self._df.loc[run_number] = values
        if len(self._df) > p.history:
            self._df = self._df.loc[self._df.index[-p.history:]]
        self._old_histo = old_histo
        return self._return_run_step(dfslot.next_state(), steps_run=steps)
