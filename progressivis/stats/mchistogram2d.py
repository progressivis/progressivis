from __future__ import absolute_import, division, print_function

from ..core.utils import indices_len, fix_loc, get_physical_base
from ..core.slot import SlotDescriptor
#from ..table.module import TableModule
from ..table.table import Table
from ..table.nary import NAry
from ..core.module import Module
from fast_histogram import histogram2d
#from timeit import default_timer
import numpy as np
import scipy as sp
import logging
from progressivis.storage import Group
logger = logging.getLogger(__name__)



class MCHistogram2D(NAry):
    parameters = [('xbins',  np.dtype(int),   256),
                  ('ybins',  np.dtype(int),   256),
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

    def __init__(self, x_column, y_column, with_output=True, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('data', type=Table, required=True),])
                         #SlotDescriptor('min', type=Table, required=False),
                         #SlotDescriptor('max', type=Table, required=False)])
        super(MCHistogram2D, self).__init__(dataframe_slot='data', **kwds)
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
        n = self.generate_table_name('MCHistogram2D')
        self._table = Table(n, #self.generate_table_name('MCHistogram2D'),
                            dshape=MCHistogram2D.schema,
                            storagegroup=Group.default_internal(n),
                            #chunks={'array': (1, 64, 64)},
#                            scheduler=self.scheduler(),
                            create=True)

    def reset(self):
        self._histo = None
        self._xedges = None
        self._yedges = None
        self.total_read = 0
        self.get_input_slot('data').reset()        
    def predict_step_size(self, duration):
        return Module. predict_step_size(self, duration)


    
    def is_ready(self):
        # If we have created data but no valid min/max, we can only wait
        if self._bounds and self.get_input_slot('data').created.any():
            return True
        return super(MCHistogram2D, self).is_ready()

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

    def get_bounds(self, run_number):
        xmin = ymin = np.inf
        xmax = ymax = -np.inf
        has_creation = False
        for name in self.inputs:
            input_slot = self.get_input_slot(name)
            if input_slot is None:
                continue
            meta = input_slot.meta
            if meta is None:
                continue
            meta, x_column, y_column = meta
            if meta == 'min':
                min_slot = input_slot
                min_slot.update(run_number)
                if min_slot.created.any():
                    has_creation = True
                min_slot.created.next()
                with min_slot.lock:
                    min_df = min_slot.data()
                    if len(min_df) > 0:
                        min_ = min_df.last()
                        xmin = min(xmin, min_[x_column])
                        ymin = min(ymin, min_[y_column])
            elif meta == 'max':
                max_slot = input_slot
                max_slot.update(run_number)
                if max_slot.created.any():
                    has_creation = True                
                max_slot.created.next()
                with max_slot.lock:
                    max_df = max_slot.data()
                    if len(max_df) > 0:
                        max_ = max_df.last()
                        xmax = max(xmax, max_[x_column])
                        ymax = max(ymax, max_[y_column])
        if xmax < xmin:
            xmax, xmin = xmin, xmax
            logger.warning('xmax < xmin, swapped')
        if ymax < ymin:
            ymax, ymin = ymin, ymax
            logger.warning('ymax < ymin, swapped')
        if np.inf in (xmin, -xmax, ymin, -ymax):
            return None
        return (xmin, xmax, ymin, ymax, has_creation)
    
    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('data')
        dfslot.update(run_number)
        if dfslot.updated.any():
            logger.debug('reseting histogram')
            self.reset()
            dfslot.update(run_number)
        bounds = self.get_bounds(run_number)
        if bounds is None:
            logger.debug('No bounds yet at run %d', run_number)
            return self._return_run_step(self.state_blocked, steps_run=0)
        xmin, xmax, ymin, ymax, has_creation = bounds
        if not (dfslot.created.any() or has_creation):
            logger.info('Input buffers empty')
            return self._return_run_step(self.state_blocked, steps_run=0)                    
        if self._bounds is None:
            (xdelta, ydelta) = self.get_delta(xmin, xmax, ymin, ymax)
            self._bounds = (xmin-xdelta,xmax+xdelta,ymin-ydelta,ymax+ydelta)
            logger.info("New bounds at run %d: %s", run_number,self._bounds)
        else:
            (dxmin, dxmax, dymin, dymax) = self._bounds
            (xdelta, ydelta) = self.get_delta(xmin, xmax, ymin, ymax)
            assert xdelta >= 0 and ydelta >= 0
            
            # Either the min/max has extended, or it has shrunk beyond the deltas
            if ((xmin<dxmin or xmax>dxmax or ymin<dymin or ymax>dymax)
                or (xmin>(dxmin+xdelta) or xmax<(dxmax-xdelta) or ymin>(dymin+ydelta) or ymax<(dymax-ydelta))):
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
        # or to update the previous if is still exists (i.e. no reset)
        p = self.params
        steps = 0
        # if there are new deletions, build the histogram of the deleted pairs
        # then subtract it from the main histogram
        if dfslot.deleted.any() and self._histo is not None:
            input_df = get_physical_base(dfslot.data())
            indices = dfslot.deleted.next(step_size)
            steps += indices_len(indices)
            #print('Histogram2D steps :%d'% steps)
            logger.info('Read %d rows', steps)
            x = input_df[self.x_column]
            y = input_df[self.y_column]
            idx = input_df.id_to_index(fix_loc(indices))
            #print(idx)
            x = x[idx]
            y = y[idx]
            bins = [p.ybins, p.xbins]
            if len(x)>0:
                histo = histogram2d(y, x,
                                    bins=bins,
                                    range=[[ymin, ymax], [xmin, xmax]])
                self._histo -= histo
        # if there are new creations, build a partial histogram with them then
        # add it to the main histogram
        input_df = dfslot.data()
        indices = dfslot.created.next(step_size)
        steps += indices_len(indices)
        #print('Histogram2D steps :%d'% steps)
        logger.info('Read %d rows', steps)
        self.total_read += steps
        
        x = input_df[self.x_column]
        y = input_df[self.y_column]
        idx = input_df.id_to_index(fix_loc(indices))
        x = x[idx]
        y = y[idx]
        if self._xedges is not None:
            bins = [self._xedges, self._yedges]
        else:
            bins = [p.ybins, p.xbins]
        if len(x)>0:
            # using fast_histogram
            histo = histogram2d(y, x,
                                bins=bins,
                                range=[[ymin, ymax], [xmin, xmax]])
        else:
            histo = None
            cmax = 0

        if self._histo is None:
            self._histo = histo
        elif histo is not None:
            self._histo += histo

        if self._histo is not None:
            cmax = self._histo.max()
        values = {'array': np.flip(self._histo, axis=0),
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

    def old_build_heatmap(self, values):
        if not values:
            return
        p = self.params
        json_ = {'columns': [self.x_column, self.y_column], 'xbins': p.xbins,
                     'ybins':p.ybins}
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
                    json_['image'] = sp.misc.bytescale(data)
                    self._heatmap_cache = json_
                    
    def build_heatmap(self, values):
        json_ = {}
        row = values
        if not (np.isnan(row['xmin']) or np.isnan(row['xmax'])
                    or np.isnan(row['ymin']) or np.isnan(row['ymax'])):
            bounds = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            data = row['array']
            #data = sp.special.cbrt(row['array'])
            #json_['data'] = sp.misc.bytescale(data)
            json_['binnedPixels'] = data
            json_['range'] = [np.min(data), np.max(data)]
            json_['count'] = np.sum(data)
            json_['value'] = "heatmap"
            #return json_
            self._heatmap_cache = (json_, bounds)
        return None
    
    def old_heatmap_to_json(self, json, short=False):
        if self._heatmap_cache:
            json.update(self._heatmap_cache)
        return json

    def heatmap_to_json(self, json, short=False):
        if self._heatmap_cache is None:
            return json
        x_label, y_label = "x", "y"
        domain = ["Heatmap"]
        count = 1
        xmin = ymin = - np.inf
        xmax = ymax = np.inf
        buff, bounds = self._heatmap_cache
        xmin, ymin, xmax, ymax = bounds #buff.pop('bounds')
        buffers = [buff]
        # TODO: check consistency among classes (e.g. same xbin, ybin etc.)
        xbins, ybins = buffers[0]['binnedPixels'].shape
        encoding = {
            "x": {
                "bin": {
                    "maxbins": xbins
                },
                "aggregate": "count",
                "field": x_label,
                "type": "quantitative",
                "scale": {
                    "domain": [
                            -7,
                        7
                    ],
                    "range": [
                        0,
                        xbins
                    ]
                }
            },
            "z": {
                "field": "category",
                "type": "nominal",
                "scale": {
                    "domain": domain
                }
            },
            "y": {
                "bin": {
                    "maxbins": ybins
                },
                "aggregate": "count",
                "field": y_label,
                "type": "quantitative",
                "scale": {
                    "domain": [
                            -7,
                        7
                    ],
                    "range": [
                        0,
                        ybins
                    ]
                }
            }
        }
        source = {"program": "progressivis",
                "type": "python",
                "rows": count
                }
        json['chart'] = dict(buffers=buffers, encoding=encoding, source=source)
        json['bounds'] = dict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        json['sample'] = dict(data=[], index=[])
        json['columns'] = [x_label, y_label]
        return json
    
    
    def is_visualization(self):
        return True

    def get_visualization(self):
        return "heatmap"

    def to_json(self, short=False):
        json = super(MCHistogram2D, self).to_json(short)
        if short:
            return json
        return self.heatmap_to_json(json, short)
