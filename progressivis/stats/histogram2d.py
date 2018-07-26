from __future__ import absolute_import, division, print_function

from progressivis.core.utils import indices_len, fix_loc
from progressivis.core.slot import SlotDescriptor
from progressivis.table.module import TableModule
from progressivis.table.table import Table
from fast_histogram import histogram2d
#from timeit import default_timer
import numpy as np
from pyfastpfor import *
import scipy as sp
import logging
logger = logging.getLogger(__name__)

COMPRESSION = 0

class Histogram2D(TableModule):
    parameters = [('xbins',  np.dtype(int),   512),
                  ('ybins',  np.dtype(int),   512),
                  ('xdelta', np.dtype(float), -5), # means 5%
                  ('ydelta', np.dtype(float), -5), # means 5%
                  ('history',np.dtype(int),   3) ]

    schema = "{" \
             "array: var * var * float64," \
             "cmin: float64," \
             "cmax: float64," \
             "xmin: float64," \
             "xmax: float64," \
             "ymin: float64," \
             "ymax: float64," \
             "time: int64" \
             "}"

    def __init__(self, x_column, y_column, with_output=False, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True),
                         SlotDescriptor('min', type=Table, required=True),
                         SlotDescriptor('max', type=Table, required=True)])
        super(Histogram2D, self).__init__(dataframe_slot='table', **kwds)
        self.x_column = x_column
        self.y_column = y_column
        self.default_step_size = 10000
        self.total_read = 0
        self._histo = None
        self._xedges = None
        self._yedges = None
        self._bounds = None
        self._with_output = with_output
        self._heatmap_cache = None
        self._table = Table(self.generate_table_name('Histogram2D'),
                            dshape=Histogram2D.schema,
                            chunks={'array': (1, 64, 64)},
#                            scheduler=self.scheduler(),
                            create=True)

    def reset(self):
        self._histo = None
        self._xedges = None
        self._yedges = None
        self.total_read = 0
        self.get_input_slot('table').reset()        

    def is_ready(self):
        # If we have created data but no valid min/max, we can only wait
        if self._bounds and self.get_input_slot('table').created.any():
            return True
        return super(Histogram2D, self).is_ready()

    def get_bounds(self, min_slot, max_slot):
        min_slot.created.next()
        with min_slot.lock:
            min_df = min_slot.data()
            if len(min_df)==0 and self._bounds is None:
                return None
            min_ = min_df.last()
            xmin = min_[self.x_column]
            ymin = min_[self.y_column]
        
        max_slot.created.next()
        with max_slot.lock:
            max_df = max_slot.data()
            if len(max_df)==0 and self._bounds is None:
                return None
            max_ = max_df.last()
            xmax = max_[self.x_column]
            ymax = max_[self.y_column]
        
        if xmax < xmin:
            xmax, xmin = xmin, xmax
            logger.warning('xmax < xmin, swapped')
        if ymax < ymin:
            ymax, ymin = ymin, ymax
            logger.warning('ymax < ymin, swapped')
        return (xmin, xmax, ymin, ymax)

    def get_delta(self, xmin, xmax, ymin, ymax):
        p = self.params
        xdelta, ydelta = p['xdelta'], p['ydelta']
        if xdelta < 0:
            dx = xmax - xmin
            xdelta = dx*xdelta/-100.0
            logger.info('xdelta is %f', xdelta)
        if ydelta < 0:
            dy = ymax - ymin
            ydelta = dy*ydelta/-100.0
            logger.info('ydelta is %f', ydelta)
        return (xdelta, ydelta)

    def run_step(self, run_number, step_size, howlong):        
        dfslot = self.get_input_slot('table')
        dfslot.update(run_number)
        min_slot = self.get_input_slot('min')
        min_slot.update(run_number)
        max_slot = self.get_input_slot('max')
        max_slot.update(run_number)
        if dfslot.updated.any() or dfslot.deleted.any():
            logger.debug('reseting histogram')
            self.reset()
            dfslot.update(run_number)

        if not (dfslot.created.any() or min_slot.created.any() or max_slot.created.any()):
            logger.info('Input buffers empty')
            return self._return_run_step(self.state_blocked, steps_run=0)
            
        bounds = self.get_bounds(min_slot, max_slot)
        if bounds is None:
            logger.debug('No bounds yet at run %d', run_number)
            return self._return_run_step(self.state_blocked, steps_run=0)
        xmin, xmax, ymin, ymax = bounds
        if self._bounds is None:
            (xdelta, ydelta) = self.get_delta(*bounds)
            self._bounds = (xmin-xdelta,xmax+xdelta,ymin-ydelta,ymax+ydelta)
            logger.info("New bounds at run %d: %s", run_number,self._bounds)
        else:
            (dxmin, dxmax, dymin, dymax) = self._bounds
            (xdelta, ydelta) = self.get_delta(*bounds)
            assert xdelta >= 0 and ydelta >= 0
            
            # Either the min/max has extended, or it has shrunk beyond the deltas
            if (xmin<dxmin or xmax>dxmax or ymin<dymin or ymax>dymax) \
              or (xmin>(dxmin+xdelta) or xmax<(dxmax-xdelta) or ymin>(dymin+ydelta) or ymax<(dymax-ydelta)):
                #print('Old bounds: %s,%s,%s,%s'%(dxmin,dxmax,dymin,dymax))
                self._bounds = (xmin-xdelta,xmax+xdelta,ymin-ydelta,ymax+ydelta)
                #print('Updated bounds at run %d: %s old %s deltas %s, %s'%(run_number,self._bounds, bounds, xdelta, ydelta))
                logger.info('Updated bounds at run %s: %s', run_number, self._bounds)
                self.reset()
                dfslot.update(run_number)


        xmin, xmax, ymin, ymax = self._bounds
        if xmin>=xmax or ymin>=ymax:
            logger.error('Invalid bounds: %s', self._bounds)
            return self._return_run_step(self.state_blocked, steps_run=0)

        # Now, we know we have data and bounds, proceed to create a new histogram

        input_df = dfslot.data()
        indices = dfslot.created.next(step_size)
        steps = indices_len(indices)
        #print('Histogram2D steps :%d'% steps)
        logger.info('Read %d rows', steps)
        self.total_read += steps
        
        x = input_df[self.x_column]
        y = input_df[self.y_column]
        idx = input_df.id_to_index(fix_loc(indices))
        #print(idx)
        x = x[idx]
        y = y[idx]
        p = self.params
        if self._xedges is not None:
            bins = [self._xedges, self._yedges]
        else:
            bins = [p.ybins, p.xbins]
        if len(x)>0:
            #t = default_timer()
            # using fast_histogram
            histo = histogram2d(y, x,
                                bins=bins,
                                range=[[ymin, ymax], [xmin, xmax]])
            # using numpy histogram
            #histo, xedges, yedges = np.histogram2d(y, x,
            #                                           bins=bins,
            #                                           range=[[ymin, ymax], [xmin, xmax]],
            #                                           normed=False)
            #t = default_timer()-t
            #print('Time for histogram2d: %f'%t)
            #self._xedges = xedges
            #self._yedges = yedges
                
        else:
            histo = None
            cmax = 0

        if self._histo is None:
            self._histo = histo
        elif histo is not None:
            self._histo += histo

        if self._histo is not None:
            cmax = self._histo.max()
        values = {'array': self._histo,
                  'cmin': 0,
                  'cmax': cmax,
                  'xmin': xmin,
                  'xmax': xmax,
                  'ymin': ymin,
                  'ymax': ymax,
                  'time': run_number}
        if self._with_output:
            with self.lock:
                table = self._table
                table['array'].set_shape([p.ybins, p.xbins])
                l = len(table)
                last = table.last()
                if l == 0 or last['time'] != run_number:
                    table.add(values)
                else:
                    table.iloc[last.row] = values
        self.build_heatmap(values)
        return self._return_run_step(self.next_state(dfslot), steps_run=steps)

    def build_heatmap(self, values):
        if not values:
            return
        p = self.params
        json_ = {'columns': [self.x_column, self.y_column], 'xbins': p.xbins,
                     'ybins':p.ybins, 'compression': COMPRESSION}
        with self.lock:
                row = values
                if not (np.isnan(row['xmin']) or np.isnan(row['xmax'])
                        or np.isnan(row['ymin']) or np.isnan(row['ymax'])):
                    json_['bounds'] = {
                        'xmin': row['xmin'],
                        'ymin': row['ymin'],
                        'xmax': row['xmax'],
                        'ymax': row['ymax']
                    }
                    data = sp.special.cbrt(row['array'])
                    data = sp.misc.bytescale(data)
                    if COMPRESSION:
                        inp = data.astype(np.uint32).flatten()
                        arrSize = len(inp)
                        inpComp = np.zeros(arrSize + 1024, dtype = np.uint32, order = 'C')
                        codec = getCodec('vbyte')
                        compSize = codec.encodeArray(inp, arrSize, inpComp, len(inpComp))
                        #print("compSize: ", compSize, "arrSize: ", arrSize)
                        #print('Compression ratio: %g' % (float(compSize)/arrSize))
                        json_['image'] = inpComp[:compSize-1]
                        self._heatmap_cache = json_
                    else:
                        json_['image'] = data
                    self._heatmap_cache = json_

    def heatmap_to_json(self, json, short=False):
        if self._heatmap_cache:
            #import pdb;pdb.set_trace()
            json.update(self._heatmap_cache)
        return json
    
    
    def is_visualization(self):
        return True

    def get_visualization(self):
        return "heatmap"

    def to_json(self, short=False):
        json = super(Histogram2D, self).to_json(short)
        if short:
            return json
        return self.heatmap_to_json(json, short)
