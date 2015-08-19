"""Visualize DataFrame columns x,y on the notebook, allowing refreshing."""
from __future__ import print_function

from progressivis import *
from progressivis.core.dataframe import DataFrameModule
from progressivis.stats import Histogram2D, Stats

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import Blues4, YlOrRd9
from bokeh.models import ColumnDataSource

from ipywidgets import widgets
from IPython.display import display
#output_notebook()

import numpy as np

class ScatterPlot(DataFrameModule):
    parameters = [('xmin',   np.dtype(float), 0),
                  ('xmax',   np.dtype(float), 1),
                  ('ymin',   np.dtype(float), 0),
                  ('ymax',   np.dtype(float), 1) ]
        
    def __init__(self, x_column, y_column, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('histograph2d', type=pd.DataFrame),
                         SlotDescriptor('df', type=pd.DataFrame) ])
        super(ScatterPlot, self).__init__(quantum=0.1, **kwds)
        self.x_column = x_column
        self.y_column = y_column

    def create_scatterplot_modules(self):
        wait = Wait(delay=2,group=self.id)
        #wait.input.inp = df_slot
        x_stats = Stats(self.x_column, min_column='xmin', max_column='xmax',group=self.id)
        x_stats.input._params = wait.output.out
        y_stats = Stats(self.y_column, min_column='ymin', max_column='ymax',group=self.id)
        y_stats.input._params = wait.output.out
        merge = Merge(group=self.id)
        merge.input.df = x_stats.output.stats
        merge.input.df = y_stats.output.stats # magic input df slot
        histogram2d = Histogram2D(self.x_column, self.y_column,group=self.id);
        histogram2d.input.df = wait.output.out
        histogram2d.input._params = merge.output.df
        self.input.histogram2d = histogram2d.output.histogram2d
        self.input.df = wait.output.out
        return wait

    def run_step(self,run_number,step_size,howlong):
        pass


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

    def update(self):
        histo_df = self.input.histogram2d.data()
        if histo_df is not None:
            self.image_source.data['image'] = [histo_df.at[histo_df.index[-1],'array']]
            self.image_source.push_notebook()
        df = self.df()
        if df is not None:
            # TODO sample
            self.scatter_source.data['x'] = df[self.x_column]
            self.scatter_source.data['y'] = df[self.y_column]
            self.scatter_source.push_notebook()
