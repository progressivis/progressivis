"Multiclass Module."
from __future__ import absolute_import, division, print_function

import numpy as np
from ..table.nary import NAry
from ..stats import MCHistogram2D, Sample
from ..table.range_query_2d import RangeQuery2d
from ..core import ProgressiveError
from ..io import Variable, VirtualVariable

from itertools import chain

from collections import defaultdict


class _DataClass(object):
    def __init__(self, name, group, x_column, y_column, approximate=False,
                 scheduler=None, **kwds):
        self.name = name
        self._group = group
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

    def scheduler(self):
        return self._scheduler

    def create_dependent_modules(self, input_module, input_slot,
                                 histogram2d=None, heatmap=None,
                                 sample=True, select=None, **kwds):
        if self.input_module is not None:
            return self
        scheduler = self.scheduler()
        with scheduler:
            self.input_module = input_module
            self.input_slot = input_slot
            range_query_2d = RangeQuery2d(column_x=self.x_column,
                                          column_y=self.y_column,
                                          group=self._group,
                                          scheduler=scheduler,
                                          approximate=self._approximate)
            range_query_2d.create_dependent_modules(input_module,
                                                    input_slot,
                                                    min_value=False,
                                                    max_value=False)
            self.min_value = Variable(group=self._group, scheduler=scheduler)
            self.min_value.input.like = range_query_2d.min.output.table
            range_query_2d.input.lower = self.min_value.output.table
            self.max_value = Variable(group=self._group, scheduler=scheduler)
            self.max_value.input.like = range_query_2d.max.output.table
            range_query_2d.input.upper = self.max_value.output.table
            if histogram2d is None:
                histogram2d = MCHistogram2D(self.x_column, self.y_column,
                                            group=self._group,
                                            scheduler=scheduler)
            histogram2d.input.data = range_query_2d.output.table
            if sample is True:
                sample = Sample(samples=100, group=self._group,
                                scheduler=scheduler)
            elif sample is None and select is None:
                raise ProgressiveError("Scatterplot needs a select module")
            if sample is not None:
                sample.input.table = range_query_2d.output.table
            self.histogram2d = histogram2d
            self.sample = sample
            self.select = select
            self.min = range_query_2d.min.output.table
            self.max = range_query_2d.max.output.table
            self.range_query_2d = range_query_2d
        scatterplot = self
        return scatterplot


class MCScatterPlot(NAry):
    "Module executing multiclass."
    def __init__(self, classes, x_label="x", y_label="y", approximate=False,
                 **kwds):
        """Multiclass ...
        """
        super(MCScatterPlot, self).__init__(**kwds)
        self._classes = classes  # TODO: check it ...
        self._x_label = x_label
        self._y_label = y_label
        self._approximate = approximate
        self._json_cache = None
        self.input_module = None
        self.input_slot = None
        self._data_class_list = []
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
        return "mcscatterplot"

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
            ret[class_].update({
                input_type: (input_slot, meta['x'], meta['y'])
            })
        return changes, ret

    def build_heatmap(self, inp, domain):
        inp_table = inp.data()
        if inp_table is None:
            return None
        with inp_table.lock:
            last = inp_table.last()
            if last is None:
                return None
            row = last.to_dict()
            data = np.copy(row['array'])
            json_ = {}
            if not (np.isnan(row['xmin']) or np.isnan(row['xmax'])
                    or np.isnan(row['ymin']) or np.isnan(row['ymax'])):
                json_['bounds'] = (row['xmin'], row['ymin'],
                                   row['xmax'], row['ymax'])
                json_['binnedPixels'] = data
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
        xbins, ybins = buffers[0]['binnedPixels'].shape
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
        for i, s in enumerate(samples):
            d = s['data']
            for row in d:
                row.append(i)
            s_data.extend(d)
        json['sample'] = dict(data=s_data, index=list(range(len(s_data))))
        json['columns'] = [self._x_label, self._y_label]
        return json

    def run_step(self, run_number, step_size, howlong):
        return self._return_run_step(self.state_blocked, steps_run=0)

    def run(self, run_number):
        super(MCScatterPlot, self).run(run_number)
        self._json_cache = self._to_json_impl()

    def to_json(self, short=False):
        if self._json_cache:
            return self._json_cache
        return self._to_json_impl(short)

    def _to_json_impl(self, short=False):
        self.image = None
        json = super(MCScatterPlot, self).to_json(short, with_speed=False)
        if short:
            return json
        return self.make_json(json)

    def create_dependent_modules(self, input_module, input_slot,
                                 sample=True, select=None, **kwds):
        self.input_module = input_module
        self.input_slot = input_slot
        with self.scheduler():
            self.min_value = VirtualVariable([self._x_label, self._y_label])
            self.max_value = VirtualVariable([self._x_label, self._y_label])
            for cl in self._classes:
                self._add_class(*cl)
            self._finalize()

    def _finalize(self):
        for dc in self._data_class_list:
            for dc2 in self._data_class_list:
                x, y = dc2.x_column, dc2.y_column
                dc.histogram2d.input['table', ('min', x, y)] = dc2.range_query_2d.output.min
                dc.histogram2d.input['table', ('max', x, y)] = dc2.range_query_2d.output.max

    def _add_class(self, name, x_column, y_column):
        if self.input_module is None or self.input_slot is None:
            raise ProgressiveError("You have to create "
                                   "the dependent modules first!")
        data_class = _DataClass(name, self.name, x_column,
                                y_column,
                                approximate=self._approximate,
                                scheduler=self._scheduler)
        data_class.create_dependent_modules(self.input_module, self.input_slot)
        col_translation = {self._x_label: x_column, self._y_label: y_column}
        hist_meta = dict(inp='hist', class_=name, **col_translation)
        sample_meta = dict(inp='sample', class_=name, **col_translation)
        self.input['table', hist_meta] = data_class.histogram2d.output.table
        self.input['table', sample_meta] = data_class.sample.output.table
        self._data_class_list.append(data_class)
        self.min_value.subscribe(data_class.min_value, col_translation)
        self.max_value.subscribe(data_class.max_value, col_translation)

    def get_starving_mods(self):
        return chain(*[(s.histogram2d, s.sample)
                       for s in self._data_class_list])
