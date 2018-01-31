"""Visualize DataFrame columns x,y on the notebook, allowing refreshing."""


from progressivis.core import SlotDescriptor, ProgressiveError
from progressivis.table import Table
from progressivis.table.select import Select #, RangeQuery
from progressivis.table.module import TableModule

from progressivis.stats import Histogram2D, Sample, Min, Max
from progressivis.vis import Heatmap

import numpy as np

import logging
logger = logging.getLogger(__name__)


class ScatterPlot(TableModule):
    parameters = [('xmin',   np.dtype(float), 0),
                  ('xmax',   np.dtype(float), 1),
                  ('ymin',   np.dtype(float), 0),
                  ('ymax',   np.dtype(float), 1) ]
        
    def __init__(self, x_column, y_column, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('heatmap', type=Table),
                         SlotDescriptor('table', type=Table),
                         SlotDescriptor('select', type=Table)])
        super(ScatterPlot, self).__init__(quantum=0.1, **kwds)
        self.x_column = x_column
        self.y_column = y_column
        self._auto_update = False
        self.image_source = None
        self.scatter_source = None
        self.image = None
        self.input_module = None
        self.input_slot = None
        self.min = None
        self.max = None
        self.histogram2d = None
        self.heatmap = None       

#    def get_data(self, name):
#        return self.get_input_slot(name).data()
        #return super(ScatterPlot, self).get_data(name)

    def table(self):
        return self.get_input_slot('table').data()

    def is_visualization(self):
        return True

    def get_visualization(self):
        return "scatterplot"

    def create_dependent_modules(self, input_module, input_slot, histogram2d=None,heatmap=None,sample=True,select=None, **kwds):
        if self.input_module is not None:
            return self
        
        s=self.scheduler()
        self.input_module = input_module
        self.input_slot = input_slot
        
        # if range_query is None:
        #     range_query = RangeQuery(group=self.id,scheduler=s)
        #     range_query.create_dependent_modules(input_module, input_slot, **kwds)
        min_ = Min(group=self.id,scheduler=s)
        max_ = Max(group=self.id,scheduler=s)
        min_.input.table = input_module.output[input_slot]
        max_.input.table = input_module.output[input_slot]
        if histogram2d is None:
            histogram2d = Histogram2D(self.x_column, self.y_column,group=self.id,scheduler=s)
        histogram2d.input.table = input_module.output[input_slot]
        histogram2d.input.min = min_.output.table
        histogram2d.input.max = max_.output.table
        if heatmap is None:
            heatmap = Heatmap(group=self.id,filename='heatmap%d.png', history=100, scheduler=s)
        heatmap.input.array = histogram2d.output.table
        if sample is True:
            sample = Sample(samples=100,group=self.id,scheduler=s)
        elif sample is None and select is None:
            raise ProgressiveError("Scatterplot needs a select module")
        if sample is not None:
            sample.input.table = input_module.output[input_slot] #select.output.df
        if select is None:
            select = Select(group=self.id,scheduler=s)
            select.input.table = input_module.output[input_slot]
            select.input.select = sample.output.select

        scatterplot=self
        scatterplot.input.heatmap = heatmap.output.heatmap
        scatterplot.input.table = input_module.output[input_slot]
        scatterplot.input.select = select.output.table

        # self.range_query = range_query
        # self.min = range_query.min
        # self.max = range_query.max
        # self.min_value = range_query.min_value
        # self.max_value = range_query.max_value
        # self.histogram2d = histogram2d
        # self.heatmap = heatmap
        # self.sample = sample
        self.select = select
        self.min = min_
        self.max = max_
        self.histogram2d = histogram2d
        self.heatmap = heatmap        

        return scatterplot

    def predict_step_size(self, duration):
        return 1

    def run_step(self,run_number,step_size,howlong):
        return self._return_run_step(self.state_blocked, steps_run=1, reads=1, updates=1)

    def to_json(self, short=False):
        self.image = None
        json = super(ScatterPlot, self).to_json(short)
        if short:
            return json
        return self.scatterplot_to_json(json, short)

    def scatterplot_to_json(self, json, short):
        with self.lock:
            select = self.get_input_slot('select').data()
            if select is not None:
                json['scatterplot'] = select.to_json(orient='split',
                                                     columns=[self.x_column,self.y_column])
            else:
                logger.debug('Select data not found')

        heatmap = self.get_input_module('heatmap')
        return heatmap.heatmap_to_json(json, short)

    def get_image(self, run_number=None):
        heatmap = self.get_input_module('heatmap')
        return heatmap.get_image(run_number)

