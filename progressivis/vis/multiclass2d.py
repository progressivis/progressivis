"Multiclass Module."
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp
from ..core.utils import (indices_len, inter_slice, fix_loc)
from ..core.bitmap import bitmap
from ..table.nary import NAry
from ..table import Table
from .scatterplot import ScatterPlot
from ..core import SlotDescriptor, ProgressiveError
from ..io import VirtualVariable
from itertools import chain

from collections import defaultdict

class _FakeInput(object):
    def __init__(self):
        self.table = None
        self.select = None

class _FakeSP(object):
    def __init__(self, name, x_column, y_column, approximate=False, scheduler=None,**kwds):
        self.name = name
        self.x_column = x_column
        self.y_column = y_column
        self._approximate = approximate
        self._scheduler = scheduler
        self.input_module = None
        self.input_slot = None
        self.min = None
        self.max = None
        self.histogram2d = None
        self.heatmap = None
        self.min_value = None
        self.max_value = None
        self.sample = None
        self.select = None
        self.range_query_2d = None
        self.input = _FakeInput()
    def scheduler(self):
        return self._scheduler

class MultiClass2D(NAry):
    "Module executing multiclass."
    def __init__(self, x_label="x", y_label="y", approximate=False, **kwds):
        """Multiclass ...
        """
        super(MultiClass2D, self).__init__(**kwds)
        self._x_label = x_label
        self._y_label = y_label
        self._approximate = approximate
        self._json_cache = None
        self.input_module = None
        self.input_slot = None
        self._fake_sp_list = []
        self.min_value = None
        self.max_value = None

    def forget_changes(self, input_slot):
        changes = False
        if input_slot.deleted.any():
            input_slot.deleted.next()
            changes = True
        if input_slot.created.any():
            input_slot.created.next()
            changes = True
        if input_slot.updated.any():
            input_slot.updated.next()
            changes = True
        return changes
    def is_visualization(self):
        return True

    def get_visualization(self):
        return "multiclass2d"

    def predict_step_size(self, duration):
        return 1

    def group_inputs(self):
        """
        Group inputs by classes using meta field on slots
        """
        ret = defaultdict(dict)
        changes = False
        for name in self.inputs:
            input_slot = self.get_input_slot(name)
            if input_slot is None:
                continue
            meta = input_slot.meta
            if meta is None:
                continue
            input_type = meta['inp']
            class_ = meta['class_']
            if input_type not in ('hist', 'sample'):
                raise ValueError('{} not in [hist, sample]'.format(input_type))
            
            changes |= self.forget_changes(input_slot)
            ret[class_].update({input_type: (input_slot, meta['x'], meta['y'])})
        return changes, ret
        
    def build_heatmap(self, inp, domain):
        inp_table = inp.data()
        if inp_table is None:
            return None
        last = inp_table.last()
        if last is None:
            return None
        row = last.to_dict()
        #xbins, ybins = row['array'].shape
        json_ = {}
        #row = values
        if not (np.isnan(row['xmin']) or np.isnan(row['xmax'])
                    or np.isnan(row['ymin']) or np.isnan(row['ymax'])):
            json_['bounds'] = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            data = row['array']
            #data = sp.special.cbrt(row['array'])
            #json_['data'] = sp.misc.bytescale(data)
            json_['data'] = data
            json_['range'] = [np.min(data), np.max(data)]
            json_['count'] = np.sum(data)
            json_['value'] = domain
            return json_
        return None
    
    def make_json(self, json):
        buffers = []
        domain = []
        samples = []
        count = 0
        #import pdb;pdb.set_trace()
        xmin = ymin = - np.inf
        xmax = ymax = np.inf
        changes, grouped_inputs = self.group_inputs()        
        for cname, inputs in grouped_inputs.items():
            hist_input = inputs['hist'][0]
            buff = self.build_heatmap(hist_input, cname)
            if buff is None:
                return json
            xmin_, ymin_, xmax_, ymax_ = buff.pop('bounds')
            xmin = max(xmin, xmin_)
            ymin = max(ymin, ymin_)
            xmax = min(xmax, xmax_)
            ymax = min(ymax, ymax_)
            buffers.append(buff)
            count += buff['count']
            domain.append(cname)
            sample_input = inputs['sample'][0]
            select = sample_input.data()
            x_column, y_column = inputs['sample'][1],  inputs['sample'][2]
            smpl = select.to_json(orient='split', columns=[x_column, y_column])
            samples.append(smpl)

        # TODO: check consistency among classes (e.g. same xbin, ybin etc.)
        xbins, ybins = buffers[0]['data'].shape
        encoding = {
            "x": {
                "bin": {
                    "maxbins": xbins
                },
                "aggregate": "count",
                "field": self._x_label,
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
                "field": self._y_label,
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
        s_data = []
        s_classes = []
        for i, s in enumerate(samples):
            d = s['data']
            for row in d:
                row.append(i)
            s_data.extend(d)
        json['sample'] = dict(data=s_data, index=list(range(len(s_data))))
        json['columns'] = [self._x_label, self._y_label]
        #import pdb;pdb.set_trace()
        return json
        
    def run_step(self, run_number, step_size, howlong):
        #if not changes:
        #    return self._return_run_step(self.state_blocked, steps_run=0)
        return self._return_run_step(self.state_blocked, steps_run=0)

    def run(self, run_number):
        super(MultiClass2D, self).run(run_number)
        self._json_cache = self._to_json_impl()
        #import pdb;pdb.set_trace()

    def to_json(self, short=False):
        if self._json_cache:
            return self._json_cache
        return self._to_json_impl(short)

    def _to_json_impl(self, short=False):
        self.image = None
        json = super(MultiClass2D, self).to_json(short, with_speed=False)
        if short:
            return json
        return self.make_json(json)

    def create_dependent_modules(self, input_module, input_slot,
                                 sample=True, select=None, **kwds):
        self.input_module = input_module
        self.input_slot = input_slot
        self.min_value = VirtualVariable([self._x_label, self._y_label])
        self.max_value = VirtualVariable([self._x_label, self._y_label])
        
    def add_class(self, name, x_column, y_column):
        if self.input_module is None or self.input_slot is None:
            raise ProgressiveError("You have to create the dependent modules first!")
        fake_sp = _FakeSP(name, x_column, y_column, approximate=self._approximate,
                              scheduler=self.scheduler())
        ScatterPlot.create_dependent_modules(fake_sp, self.input_module, self.input_slot)
        col_translation = {self._x_label: x_column, self._y_label: y_column}
        hist_meta = dict(inp='hist', class_=name, **col_translation)
        sample_meta = dict(inp='sample', class_=name, **col_translation)
        self.input['table', hist_meta] = fake_sp.histogram2d.output.table
        self.input['table', sample_meta] = fake_sp.sample.output.table
        self._fake_sp_list.append(fake_sp)
        self.min_value.subscribe(fake_sp.min_value, col_translation)
        self.max_value.subscribe(fake_sp.max_value, col_translation)

    def get_starving_mods(self):
        return chain(*[(s.histogram2d, s.sample) for s in self._fake_sp_list])
