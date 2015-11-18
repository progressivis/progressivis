"""Visualize DataFrame columns x,y on the notebook, allowing refreshing."""
from __future__ import print_function

from progressivis import SlotDescriptor, Wait, Join, Select
from progressivis.core.dataframe import DataFrameModule
from progressivis.stats import Histogram2D, Stats, Sample
from progressivis.vis import Heatmap

from bokeh.plotting import show
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import YlOrRd9
from bokeh.models import ColumnDataSource, Range1d

from ipywidgets import widgets
from IPython.display import display
#output_notebook()

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class ScatterPlot(DataFrameModule):
    parameters = [('xmin',   np.dtype(float), 0),
                  ('xmax',   np.dtype(float), 1),
                  ('ymin',   np.dtype(float), 0),
                  ('ymax',   np.dtype(float), 1) ]
        
    def __init__(self, x_column, y_column, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('heatmap', type=pd.DataFrame),
                         SlotDescriptor('df', type=pd.DataFrame) ])
        super(ScatterPlot, self).__init__(quantum=0.1, **kwds)
        self.x_column = x_column
        self.y_column = y_column
        self._auto_update = False
        self.image_source = None
        self.scatter_source = None
        self.image = None
#        self.bounds_source = None

    def df(self):
        return self.get_input_slot('df').data()

    def is_visualization(self):
        return True

    def get_visualization(self):
        return "scatterplot";

    def create_scatterplot_modules(self, wait=None, x_stats=None, y_stats=None, select=None, sample=None, join=None, histogram2d=None):
        s=self._scheduler
        if wait is None:
            wait = Wait(reads=0,group=self.id,scheduler=s)
        if x_stats is None:
            x_stats = Stats(self.x_column, min_column='xmin', max_column='xmax',group=self.id,scheduler=s)
        x_stats.input.df = wait.output.out
        if y_stats is None:
            y_stats = Stats(self.y_column, min_column='ymin', max_column='ymax',group=self.id,scheduler=s)
        y_stats.input.df = wait.output.out
        if join is None:
            join = Join(group=self.id,scheduler=s)
        join.input.df = x_stats.output.stats
        join.input.df = y_stats.output.stats # magic input df slot
        if select is None:
            select = Select(group=self.id,scheduler=s)
        select.input.df = wait.output.out
        if histogram2d is None:
            histogram2d = Histogram2D(self.x_column, self.y_column,group=self.id,scheduler=s);
        histogram2d.input.df = select.output.df
        histogram2d.input._params = join.output.df
        heatmap = Heatmap(group=self.id,filename='heatmap%d.png', history=100, scheduler=s)
        heatmap.input.array = histogram2d.output.histogram2d
        if sample is None:
            sample = Sample(n=500,group=self.id,scheduler=s)
        sample.input.df = select.output.df

        self.wait = wait
        self.select = select
        self.x_stats = x_stats
        self.y_stats = y_stats
        self.join = join
        self.histogram2d = histogram2d
        self.heatmap = heatmap
        self.sample = sample
        
        self.input.heatmap = heatmap.output.heatmap
        self.input.df = sample.output.sample
        return wait

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
        df = self.df()
        if df is not None:
            json['scatterplot'] = self.remove_nan(df[[self.x_column,self.y_column]]
                                                  .to_dict(orient='split'))

        heatmap = self.get_input_module('heatmap')
        return heatmap.heatmap_to_json(json, short)

    def get_image(self, run_number=None):
        heatmap = self.get_input_module('heatmap')
        return heatmap.get_image(run_number)

    # For Bokeh, but not ready for prime time yet...
    x = np.array([0, 10, 50, 90, 100], np.dtype(float))
    y = np.array([0, 50, 90, 10, 100], np.dtype(float))
    img = np.zeros((3, 3), np.float)

    def show(self, p):
        self.figure = p
        self.image_source = ColumnDataSource(data={
            'image': [self.img],
            'x': [0],
            'y': [0],
            'dw': [100],
            'dh': [100]})
        self.palette = YlOrRd9[::-1]
        p.image(image='image', x='x', y='y', dw='dw', dh='dh',
                color_mapper=LinearColorMapper(self.palette), source=self.image_source)
        self.scatter_source = ColumnDataSource(data={'x': self.x, 'y': self.y})
        p.scatter('x','y',source=self.scatter_source)
        show(self.figure)
        button = widgets.Button(description="Refresh!")
        display(button)
        button.on_click(self.update)

    def update(self, b):
        if self.image_source is None:
            return
        logger.info("Updating module '%s.%s'", self.pretty_typename(), self.id)
        #TODO use data from the same run
        histo_df = self.get_input_slot('histogram2d').data()
        row = None
        df = self.df()
        if df is not None:
            self.scatter_source.data['x'] = df[self.x_column]
            self.scatter_source.data['y'] = df[self.y_column]
            self.scatter_source.push_notebook()

        if histo_df is not None and histo_df.index[-1] is not None:
            idx = histo_df.index[-1]
            row = histo_df.loc[idx]
            if not (np.isnan(row.xmin) or np.isnan(row.xmax)
                    or np.isnan(row.ymin) or np.isnan(row.ymax)
                    or row.array is None):
                self.image_source.data['image'] = [row.array]
                self.image_source.data['x'] = [row.xmin]
                self.image_source.data['y'] = [row.ymin]
                self.image_source.data['dw'] = [row.xmax-row.xmin]
                self.image_source.data['dh'] = [row.ymax-row.ymin]
                self.image_source.push_notebook()
                self.figure.set(x_range=Range1d(row.xmin, row.xmax),
                                y_range=Range1d(row.ymin, row.ymax))
                logger.debug('Bounds: %g,%g,%g,%g', row.xmin, row.xmax, row.ymin, row.ymax)
            else:
                logger.debug('Cannot compute bounds from image')

    @property
    def auto_update(self):
        return self._auto_update
    @auto_update.setter
    def auto_update(self, value):
        self._auto_update = value

    def cleanup_run(self, run_number):
        super(ScatterPlot, self).cleanup_run(run_number)
        if self._auto_update:
            self.update(None)

