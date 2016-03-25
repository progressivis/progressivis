from progressivis.core.utils import indices_len, create_dataframe, last_row, fix_loc
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

class Histogram2D(DataFrameModule):
    parameters = [('xbins',  np.dtype(int),   512),
                  ('ybins',  np.dtype(int),   512),
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
        self._histo = None
        self._xedges = None
        self._yedges = None
        self._bounds = None
        self._df = create_dataframe(Histogram2D.schema)

    def is_ready(self):
        # If we have created data but no valid min/max, we can only wait
        if self._bounds and self.get_input_slot('df').has_created():
            return True
        return super(Histogram2D, self).is_ready()

    def get_bounds(self, min_slot, max_slot):
        min_slot.next_created()
        with min_slot.lock:
            min_df = min_slot.data()
            if len(min_df)==0 and self._bounds is None:
                return None
            min = last_row(min_df)
            xmin = min[self.x_column]
            ymin = min[self.y_column]
        
        max_slot.next_created()
        with max_slot.lock:
            max_df = max_slot.data()
            if len(max_df)==0 and self._bounds is None:
                return None
            max = last_row(max_df)
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

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)
        min_slot = self.get_input_slot('min')
        min_slot.update(run_number)
        max_slot = self.get_input_slot('max')
        max_slot.update(run_number)

        if dfslot.has_updated() or dfslot.has_deleted():
            logger.debug('reseting histogram')
            dfslot.reset()
            self._histo = None
            self._xedges = None
            self._yedges = None
            dfslot.update(run_number)

        if not (dfslot.has_created() or min_slot.has_created() or max_slot.has_created()):
            # nothing to do, just wait 
            logger.info('Input buffers empty')
            return self._return_run_step(self.state_blocked, steps_run=0)
            
        bounds = self.get_bounds(min_slot, max_slot)
        if bounds is None:
            print('No bounds yet at run %d'%run_number)
            logger.debug('No bounds yet at run %d', run_number)
            return self._return_run_step(self.state_blocked, steps_run=0)
        xmin, xmax, ymin, ymax = bounds
        if self._bounds is None:
            (xdelta, ydelta) = self.get_delta(*bounds)
            self._bounds = (xmin-xdelta,xmax+xdelta,ymin-ydelta,ymax+ydelta)
            print('New bounds at run %d: %s'%(run_number,self._bounds))
        else:
            (dxmin, dxmax, dymin, dymax) = self._bounds
            (xdelta, ydelta) = self.get_delta(*bounds)
            # Either the min/max has extended, or it has shrunk beyond the deltas
            if (xmin<dxmin or xmax>dxmax or ymin<dymin or ymax>dymax) \
              or (xmin>(dxmin+xdelta) or xmax<(dxmax-xdelta) or ymin>(dymin+ydelta) or ymax<(dymax-ydelta)):
                self._bounds = (xmin-xdelta,xmax+xdelta,ymin-ydelta,ymax+ydelta)
                print('Updated bounds at run %d: %s'%(run_number,self._bounds))
                logger.info('Updated bounds at run %s: %s', run_number, self._bounds)
                dfslot.reset()
                dfslot.update(run_number) # should recompute the histogram from scatch
                self._histo = None 
                self._xedges = None
                self._yedges = None

        xmin, xmax, ymin, ymax = self._bounds
        if xmin>=xmax or ymin>=ymax:
            logger.error('Invalid bounds: %s', self._bounds)
            return self._return_run_step(self.state_blocked, steps_run=0)

        # Now, we know we have data and bounds, proceed to create a new histogram

        input_df = dfslot.data()
        indices = dfslot.next_created(step_size)
        steps = indices_len(indices)
        logger.info('Read %d rows', steps)
        self.total_read += steps
        
        filtered_df = input_df.loc[fix_loc(indices)]
        x = filtered_df[self.x_column]
        y = filtered_df[self.y_column]
        p = self.params
        if self._xedges is not None:
            bins = [self._xedges, self._yedges]
        else:
            bins = [p.ybins, p.xbins]
        if len(x)>0:
            histo, xedges, yedges = np.histogram2d(y, x,
                                                   bins=bins,
                                                   range=[[ymin, ymax], [xmin, xmax]],
                                                   normed=False)
            self._xedges = xedges
            self._yedges = yedges
        else:
            histo = None
            cmax = 0

        if self._histo is None:
            self._histo = histo
        elif histo is not None:
            self._histo += histo

        if self._histo is not None:
            cmax = self._histo.max()
        print 'cmax=%d'%cmax
        values = [self._histo, 0, cmax, xmin, xmax, ymin, ymax, run_number]
        with self.lock:
            self._df.loc[run_number] = values
            if len(self._df) > p.history:
                self._df = self._df.loc[self._df.index[-p.history:]]
        return self._return_run_step(dfslot.next_state(), steps_run=steps)
