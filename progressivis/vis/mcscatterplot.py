"Multiclass Scatterplot Module."

import numpy as np
from ..table.nary import NAry
from ..stats import MCHistogram2D, Sample
from ..table.range_query_2d import RangeQuery2d
from ..utils.errors import ProgressiveError
from ..core.utils import is_notebook, get_physical_base
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
        self.range_query_2d = None

    def scheduler(self):
        return self._scheduler

    def create_dependent_modules(self, input_module, input_slot,
                                 histogram2d=None, heatmap=None,
                                 **kwds):
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
            self.min_value.input.like = range_query_2d.min.output.result
            range_query_2d.input.lower = self.min_value.output.result
            self.max_value = Variable(group=self._group, scheduler=scheduler)
            self.max_value.input.like = range_query_2d.max.output.result
            range_query_2d.input.upper = self.max_value.output.result
            if histogram2d is None:
                histogram2d = MCHistogram2D(self.x_column, self.y_column,
                                            group=self._group,
                                            scheduler=scheduler)
            histogram2d.input.data = range_query_2d.output.result
            if self.sample == 'default':
                self.sample = Sample(samples=100, group=self._group,
                                     scheduler=scheduler)
            if isinstance(self.sample, Sample):
                self.sample.input.table = range_query_2d.output.result
            self.histogram2d = histogram2d
            # self.sample = sample
            # self.select = select
            self.min = range_query_2d.min.output.result
            self.max = range_query_2d.max.output.result
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
        self._data_class_dict = {}
        self.min_value = None
        self.max_value = None
        self._ipydata = is_notebook()
        self.hist_tensor = None
        self.sample_tensor = None

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

    def build_heatmap(self, inp, domain, plan):
        inp_table = inp.data()
        if inp_table is None:
            return None
        #with inp_table.lock:
        last = inp_table.last()
        if last is None:
            return None
        row = last.to_dict()
        json_ = {}
        if not (np.isnan(row['xmin']) or np.isnan(row['xmax'])
                or np.isnan(row['ymin']) or np.isnan(row['ymax'])):
            data = row['array']
            json_['bounds'] = (row['xmin'], row['ymin'],
                               row['xmax'], row['ymax'])
            if self._ipydata:
                assert isinstance(plan, int)
                json_['binnedPixels'] = plan
                self.hist_tensor[:,:,plan] = row['array']
            else:
                data = np.copy(row['array'])
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
        z = len(grouped_inputs)
        if self._ipydata and self.hist_tensor is None:
            for sl in grouped_inputs.values():
                hi = sl['hist'][0]
                xbins = hi.output_module.params.xbins
                ybins = hi.output_module.params.ybins
                self.hist_tensor = np.zeros((xbins, ybins, z), dtype='int32')
                break
        for i, (cname, inputs) in enumerate(grouped_inputs.items()):
            hist_input = inputs['hist'][0]
            buff = self.build_heatmap(hist_input, cname, i)
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
            if 'sample' in inputs:
                sample_input = inputs['sample'][0]
                select = sample_input.data()
                x_column, y_column = inputs['sample'][1],  inputs['sample'][2]
            else:
                select = None

            if self._ipydata:
                smpl = []
                if select is not None:
                    ph_x = get_physical_base(select[x_column])
                    ph_y = get_physical_base(select[y_column])
                    smpl = ph_x.loc[select[x_column].index.index],  ph_y.loc[select[y_column].index.index]
                else:
                    smpl = [], []
                    
            else:
                smpl = select.to_json(orient='split', columns=[x_column, y_column]) if select is not None else []
            samples.append(smpl)
        if self._ipydata:
            samples_counter = []
            for vx, vy in samples:
                len_s = len(vx)
                assert len_s == len(vy)
                samples_counter.append(len_s)
            nsam = max(samples_counter)
            self.sample_tensor = np.zeros((nsam, 2, z), dtype='float32')
            for i, (vx, vy) in enumerate(samples):
                if not len(vx):
                    continue
                self.sample_tensor[:,0,i] = vx
                self.sample_tensor[:,1,i] = vy
            json['samples_counter'] = samples_counter
            samples = []
        # TODO: check consistency among classes (e.g. same xbin, ybin etc.)
        xbins, ybins = self.hist_tensor.shape[:-1] if self._ipydata else buffers[0]['binnedPixels'].shape
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
            if not s: continue
            d = s['data']
            for row in d:
                row.append(i)
            s_data.extend(d)
        json['sample'] = dict(data=s_data, index=list(range(len(s_data))))
        json['columns'] = [self._x_label, self._y_label]
        if self._ipydata:
            json['hist_tensor'] = self.hist_tensor 
            json['sample_tensor'] = self.sample_tensor
        return json

    def run_step(self, run_number, step_size, howlong):
        for name in self.get_input_slot_multiple():
            slot = self.get_input_slot(name)
            # slot.update(run_number)
            if slot.has_buffered():
                slot.clear_buffers()
                # slot.created.next()
                # slot.updated.next()
                # slot.deleted.next()
                #print("SLOT has buffered", slot)
                self._json_cache = None                
        return self._return_run_step(self.state_blocked, steps_run=0)

    def run(self, run_number):
        super(MCScatterPlot, self).run(run_number)
        if self._ipydata:
            return
        if self._json_cache is not None:
            return
        self._json_cache = self._to_json_impl()

    def to_json(self, short=False):
        if self._json_cache:
            return self._json_cache
        self._json_cache = self._to_json_impl(short)
        return self._json_cache

    def _to_json_impl(self, short=False):
        self.image = None
        json = super(MCScatterPlot, self).to_json(short, with_speed=False)
        if short:
            return json
        return self.make_json(json)

    def create_dependent_modules(self, input_module=None, input_slot=None,
                                 sample='default', **kwds):
        self.input_module = input_module
        self.input_slot = input_slot
        scheduler = self.scheduler()
        with scheduler:
            self.min_value = VirtualVariable([self._x_label, self._y_label],
                                             scheduler=scheduler)
            self.max_value = VirtualVariable([self._x_label, self._y_label],
                                             scheduler=scheduler)
            for cl in self._classes:
                if isinstance(cl, tuple):
                    self._add_class(*cl)
                else:
                    self._add_class(**cl)
            self._finalize()

    def __getitem__(self, key):
        return self._data_class_dict[key]

    def _finalize(self):
        for dc in self._data_class_dict.values():
            for dc2 in self._data_class_dict.values():
                x, y = dc2.x_column, dc2.y_column
                dc.histogram2d.input['table', ('min', x, y)] = dc2.range_query_2d.output.min
                dc.histogram2d.input['table', ('max', x, y)] = dc2.range_query_2d.output.max

    def _add_class(self, name, x_column, y_column, sample='default', sample_slot='result', input_module=None, input_slot=None):
        if self.input_module is None and input_module is None:
            raise ProgressiveError("Input module is not defined!")            
        if self.input_module is not None and input_module is not None:
            raise ProgressiveError("Input module is defined twice!")            
        if self.input_slot is None and input_slot is None:
            raise ProgressiveError("Input slot is not defined!")            
        if self.input_slot is not None and input_slot is not None:
            raise ProgressiveError("Input slot is defined twice!")            
        data_class = _DataClass(name, self.name, x_column,
                                y_column,
                                approximate=self._approximate,
                                scheduler=self._scheduler)
        data_class.sample = sample
        input_module = input_module or self.input_module
        input_slot = input_slot or self.input_slot
        data_class.create_dependent_modules(input_module, input_slot)
        col_translation = {self._x_label: x_column, self._y_label: y_column}
        hist_meta = dict(inp='hist', class_=name, **col_translation)
        self.input['table', hist_meta] = data_class.histogram2d.output.result
        if sample is not None:
            sample_meta = dict(inp='sample', class_=name, **col_translation)        
            self.input['table', sample_meta] = data_class.sample.output[sample_slot]
        self._data_class_dict[name] = data_class
        self.min_value.subscribe(data_class.min_value, col_translation)
        self.max_value.subscribe(data_class.max_value, col_translation)

    def get_starving_mods(self):
        return chain(*[(s.histogram2d, s.sample)
                       for s in self._data_class_dict.values()])
