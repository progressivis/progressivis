"""Visualize DataFrame columns x,y on the notebook, allowing refreshing."""

from ..io import Variable

from progressivis.core import SlotDescriptor, ProgressiveError
from progressivis.table import Table
from progressivis.table.select import Select #, RangeQuery
from progressivis.table.module import TableModule

from progressivis.stats import Histogram2D, Sample, Min, Max
from progressivis.vis import Heatmap
from ..table.range_query import RangeQuery
from ..table.hist_index import HistogramIndex
from ..table.bisectmod import Bisect, _get_physical_table
from ..table.intersection import Intersection
import numpy as np

import logging
logger = logging.getLogger(__name__)

WITH_INTERSECTION = False

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

    def seq_make_range_query(self, input_module, input_slot, column, min_, max_, min_value, max_value, group, scheduler):
        """ 
        Not dropped yet cause potentially useful for debugging TableSelectedView
        """
        hist_index = HistogramIndex(column=column, group=group, scheduler=scheduler)
        hist_index.input.table = input_module.output[input_slot]
        hist_index.input.min = min_.output.table
        hist_index.input.max = max_.output.table
        bisect_min = Bisect(column=column, limit_key=column,
                            op='>=',
                            hist_index=hist_index,
                            group=group,
                            scheduler=scheduler)
        bisect_min.input.table = hist_index.output.table
        bisect_min.input.limit = min_value.output.table
        bisect_max = Bisect(column=column, limit_key=column,
                            op='<=',
                            hist_index=hist_index,
                            group=group,
                            scheduler=scheduler)
        bisect_max.input.table = bisect_min.output.table
        bisect_max.input.limit = max_value.output.table
        return bisect_max

    def make_range_query1d(self, input_module, input_slot, column, min_, max_, scheduler):
        hist_index = HistogramIndex(column=column, group=self.id, scheduler=scheduler)
        hist_index.input.table = input_module.output[input_slot]
        hist_index.input.min = min_.output.table
        hist_index.input.max = max_.output.table
        bisect_min = Bisect(column=column, limit_key=column,
                            op='>=',
                            hist_index=hist_index,
                            group=self.id,
                            scheduler=scheduler)
        bisect_min.input.table = hist_index.output.table
        bisect_min.input.limit = self.min_value.output.table
        bisect_max = Bisect(column=column, limit_key=column,
                            op='<=',
                            hist_index=hist_index,
                            group=self.id,
                            scheduler=scheduler)
        bisect_max.input.table = bisect_min.output.table
        bisect_max.input.limit = self.max_value.output.table
        return bisect_min, bisect_max

    def make_range_query2d(self, input_module, input_slot, min_, max_, scheduler):
        min_x, max_x= self.make_range_query1d(input_module, input_slot,
                                                self.x_column, min_=min_,
                                                max_=max_, scheduler=scheduler)
        min_y, max_y= self.make_range_query1d(input_module, input_slot,
                                                self.y_column, min_=min_,
                                                max_=max_, scheduler=scheduler)
        range_query2d = Intersection(group=self.id, scheduler=scheduler)
        range_query2d.input.table = min_x.output.table
        range_query2d.input.table = max_x.output.table
        range_query2d.input.table = min_y.output.table
        range_query2d.input.table = max_y.output.table
        return range_query2d
    
    
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
        self.min_value = Variable(group=self.id, scheduler=s)
        self.min_value.input.like = min_.output.table
        self.max_value = Variable(group=self.id, scheduler=s)
        self.max_value.input.like = max_.output.table
        if not WITH_INTERSECTION:
            # RangeQuery variant
            range_query_x = RangeQuery(column=self.x_column, group=self.id,scheduler=s)
            range_query_x.create_dependent_modules(input_module, input_slot, min_=min_, max_=max_, min_value=self.min_value, max_value=self.max_value)
            range_query_y = RangeQuery(column=self.y_column, group=self.id,scheduler=s)
            range_query_y.create_dependent_modules(range_query_x, input_slot, min_=min_, max_=max_, min_value=self.min_value, max_value=self.max_value)
            range_query2d = range_query_y
        else:
            # Intersection variant
            range_query2d = self.make_range_query2d(input_module, input_slot, min_=min_, max_=max_, scheduler=s)
    
        # Sequential variant
        #range_query_x = self.seq_make_range_query(input_module, input_slot, self.x_column, min_=min_, max_=max_, min_value=self.min_value, max_value=self.max_value, group=self.id, scheduler=s)
        #range_query_y = self.seq_make_range_query(range_query_x, input_slot, self.y_column, min_=min_, max_=max_, min_value=self.min_value, max_value=self.max_value, group=self.id, scheduler=s)        
        
        
        select_output = range_query2d.output
        min_rq = Min(group=self.id,scheduler=s)
        max_rq = Max(group=self.id,scheduler=s)
        min_rq.input.table = select_output.table
        max_rq.input.table = select_output.table
        if histogram2d is None:
            histogram2d = Histogram2D(self.x_column, self.y_column,group=self.id,scheduler=s)
        histogram2d.input.table = select_output.table
        histogram2d.input.min = min_rq.output.table
        histogram2d.input.max = max_rq.output.table
        if heatmap is None:
            heatmap = Heatmap(group=self.id,filename='heatmap%d.png', history=100, scheduler=s)
        heatmap.input.array = histogram2d.output.table
        if sample is True:
            sample = Sample(samples=100,group=self.id,scheduler=s)
        elif sample is None and select is None:
            raise ProgressiveError("Scatterplot needs a select module")
        if sample is not None:
            sample.input.table =  select_output.table
        if select is None:
            select = Select(group=self.id,scheduler=s)
            select.input.table = range_query2d.output.table #input_module.output[input_slot]
            select.input.select = sample.output.select

        scatterplot=self
        scatterplot.input.heatmap = heatmap.output.heatmap
        scatterplot.input.table = input_module.output[input_slot]
        scatterplot.input.select = select.output.table

        # self.range_query = range_query
        # self.min = range_query.min
        # self.max = range_query.max
        #self.min_value = range_query.min_value
        #self.max_value = range_query.max_value
        self.histogram2d = histogram2d
        self.heatmap = heatmap
        self.sample = sample
        self.select = select
        self.min = min_
        self.max = max_
        self.histogram2d = histogram2d
        self.heatmap = heatmap        

        return scatterplot

    def predict_step_size(self, duration):
        return 1

    def run_step(self,run_number,step_size,howlong):
        print("SP step_size:", step_size)
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

