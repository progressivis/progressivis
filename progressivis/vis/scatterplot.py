"""Visualize DataFrame columns x,y on the notebook, allowing refreshing."""
from __future__ import print_function

from progressivis import *
from progressivis.core.dataframe import DataFrameModule
from progressivis.stats import Histogram2D, Stats, Sample

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import Blues4, YlOrRd9
from bokeh.models import ColumnDataSource

from ipywidgets import widgets
from IPython.display import display
#output_notebook()

import numpy as np
import pandas as pd

class ScatterPlot(DataFrameModule):
    parameters = [('xmin',   np.dtype(float), 0),
                  ('xmax',   np.dtype(float), 1),
                  ('ymin',   np.dtype(float), 0),
                  ('ymax',   np.dtype(float), 1) ]
        
    def __init__(self, x_column, y_column, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('histogram2d', type=pd.DataFrame),
                         SlotDescriptor('df', type=pd.DataFrame) ])
        super(ScatterPlot, self).__init__(quantum=0.1, **kwds)
        self.x_column = x_column
        self.y_column = y_column

    def create_scatterplot_modules(self):
        wait = Wait(delay=2,group=self.id)
        x_stats = Stats(self.x_column, min_column='xmin', max_column='xmax',group=self.id)
        x_stats.input.df = wait.output.out
        y_stats = Stats(self.y_column, min_column='ymin', max_column='ymax',group=self.id)
        y_stats.input.df = wait.output.out
        merge = Merge(group=self.id)
        merge.input.df = x_stats.output.stats
        merge.input.df = y_stats.output.stats # magic input df slot
        histogram2d = Histogram2D(self.x_column, self.y_column,group=self.id);
        histogram2d.input.df = wait.output.out
        histogram2d.input._params = merge.output.df
        sample = Sample(n=500,group=self.id)
        sample.input.df = wait.output.out
        
        self.wait = wait
        self.x_stats = x_stats
        self.y_stats = y_stats
        self.merge = merge
        self.histogram2d = histogram2d
        self.sample = sample
        
        self.input.histogram2d = histogram2d.output.histogram2d
        self.input.df = sample.output.sample
        return wait

    def run_step(self,run_number,step_size,howlong):
        return self._return_run_step(self.state_blocked, steps_run=1)


    x = np.array([0, 10, 50, 90, 100], np.dtype(float))
    y = np.array([0, 50, 90, 10, 100], np.dtype(float))
    img = np.empty((3, 3), np.float)

    def show(self, p):
        self.figure = p
        self.image_source = ColumnDataSource(data=dict(image=self.img))
        self.palette = YlOrRd9[::-1]
        p.image(image=[self.img], x=[0], y=[0], dw=[100], dh=[100], color_mapper=LinearColorMapper(self.palette), source=self.image_source)
        self.scatter_source = ColumnDataSource(data={'x': self.x, 'y': self.y})
        p.circle(x='x',y='y',source=self.scatter_source)
        show(self.figure)
        button = widgets.Button(description="Refresh!")
        display(button)
        button.on_click(self.update)

    def update(self, b):
        histo_df = self.get_input_slot('histogram2d').data()
        if histo_df is not None and histo_df.index[-1] is not None:
            idx = histo_df.index[-1]
            row = histo_df.loc[idx]
            self.image_source.data['image'] = [row.array]
            self.image_source.data['x'] = [row.xmin]
            self.image_source.data['y'] = [row.ymin]
            self.image_source.data['dw'] = [row.xmax-row.xmin]
            self.image_source.data['dh'] = [row.ymax-row.ymin]
            self.image_source.push_notebook()
        df = self.get_input_slot('df').data()
        if df is not None:
            # TODO sample
            self.scatter_source.data['x'] = df[self.x_column]
            self.scatter_source.data['y'] = df[self.y_column]
            self.scatter_source.push_notebook()
