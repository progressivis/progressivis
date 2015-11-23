import unittest

from progressivis import *
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.vis import PanZoom
from progressivis.datasets import get_dataset

import pandas as pd

class TestPanZoom(unittest.TestCase):
    def test_pan_zoom(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('smallfile'), index_col=False,header=None, scheduler=s)
        x_stats = Stats(1, min_column='xmin', max_column='xmax',scheduler=s)
        x_stats.input.df = csv.output.df
        y_stats = Stats(2, min_column='ymin', max_column='ymax',scheduler=s)
        y_stats.input.df = csv.output.df
        join = Join(scheduler=s)
        join.input.df = x_stats.output.stats
        join.input.df = y_stats.output.stats # magic input df slot
        pan_zoom=PanZoom(id='test_pan_zoom', scheduler=s)
        pan_zoom.input.bounds = join.output.df
        pan_zoom.describe()
        pr=Print(id='print', scheduler=s)
        pr.input.df = pan_zoom.output.panzoom
        s.start()

    def test_pan_zoom2(self):
        s=Scheduler()
        csv = CSVLoader(get_dataset('smallfile'), index_col=False,header=None, scheduler=s)
        x_stats = Stats(1, min_column='xmin', max_column='xmax',scheduler=s)
        x_stats.input.df = csv.output.df
        y_stats = Stats(2, min_column='ymin', max_column='ymax',scheduler=s)
        y_stats.input.df = csv.output.df
        join = Join(scheduler=s)
        join.input.df = x_stats.output.stats
        join.input.df = y_stats.output.stats # magic input df slot
        viewport = pd.DataFrame({'xmin': [0], 'xmax': [1], 'ymin': [0], 'ymax': [1]})
        cst = Constant(df=viewport, scheduler=s)
        pan_zoom=PanZoom(id='test_pan_zoom', scheduler=s)
        pan_zoom.input.bounds = join.output.df
        pan_zoom.input.viewport = cst.output.df
        pan_zoom.describe()
        pr=Print(id='print', scheduler=s)
        pr.input.df = pan_zoom.output.panzoom
        s.start()
        df=pan_zoom.df()
        res=df.loc[df.index[-1]]
        self.assertEqual(res.xmin, 0)
        self.assertEqual(res.xmax, 1)
        self.assertEqual(res.ymin, 0)
        self.assertEqual(res.ymax, 1)


if __name__ == '__main__':
    unittest.main()
