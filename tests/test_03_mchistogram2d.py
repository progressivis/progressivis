from . import ProgressiveTest
from progressivis.core import aio
from progressivis import Every
from progressivis.io import CSVLoader
from progressivis.stats import RandomTable
from progressivis.stats import MCHistogram2D, Min, Max
from progressivis.vis import Heatmap
from progressivis.datasets import get_dataset
from progressivis.table.stirrer import Stirrer, StirrerView
import pandas as pd
import numpy as np
import fast_histogram as fh

class TestMCHistogram2D(ProgressiveTest):

    def test_histogram2d(self):
        s = self.scheduler()
        csv = CSVLoader(get_dataset('bigfile'),
                        index_col=False,
                        header=None,
                        scheduler=s)
        min_ = Min(scheduler=s)
        min_.input.table = csv.output.table
        max_ = Max(scheduler=s)
        max_.input.table = csv.output.table
        histogram2d = MCHistogram2D('_1', '_2', xbins=100, ybins=100,
                                  scheduler=s)  # columns are called 1..30
        histogram2d.input.data = csv.output.table
        histogram2d.input['table', ('min', '_1', '_2')] = min_.output.table
        histogram2d.input['table', ('max', '_1', '_2')] = max_.output.table
        heatmap = Heatmap(filename='histo_%03d.png', scheduler=s)
        heatmap.input.array = histogram2d.output.table
        pr = Every(proc=self.terse, scheduler=s)
        pr.input.df = csv.output.table
        aio.run(csv.scheduler().start())
        s = histogram2d.trace_stats()

        
    def test_histogram2d1(self):
        s = self.scheduler()
        csv = CSVLoader(get_dataset('bigfile'),
                        index_col=False,
                        header=None,
                        scheduler=s)
        min_ = Min(scheduler=s)
        min_.input.table = csv.output.table
        max_ = Max(scheduler=s)
        max_.input.table = csv.output.table
        histogram2d = MCHistogram2D('_1', '_2', xbins=100, ybins=100,
                                  scheduler=s)  # columns are called 1..30
        histogram2d.input.data = csv.output.table
        histogram2d.input['table', ('min', '_1', '_2')] = min_.output.table
        histogram2d.input['table', ('max', '_1', '_2')] = max_.output.table
        heatmap = Heatmap(filename='histo_%03d.png', scheduler=s)
        heatmap.input.array = histogram2d.output.table
        pr = Every(proc=self.terse, scheduler=s)
        pr.input.df = csv.output.table
        aio.run(csv.scheduler().start())
        last = histogram2d._table.last().to_dict()
        h1 = last['array']
        bounds =  [[last['ymin'], last['ymax']], [last['xmin'], last['xmax']]]
        df = pd.read_csv(get_dataset('bigfile'), header=None, usecols=[1, 2])
        v = df.to_numpy() #.reshape(-1, 2)
        bins = [histogram2d.params.ybins, histogram2d.params.xbins]
        h2 = fh.histogram2d(v[:,1], v[:,0], bins=bins, range=bounds)
        h2 = np.flip(h2, axis=0)
        self.assertTrue(np.allclose(h1, h2))

    def t_histogram2d_impl(self, **kw):
        s = self.scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(update_column='_2',
                          fixed_step_size=1000, scheduler=s, **kw)
        stirrer.input.table = random.output.table
        min_ = Min(scheduler=s)
        min_.input.table = stirrer.output.table
        max_ = Max(scheduler=s)
        max_.input.table = stirrer.output.table
        histogram2d = MCHistogram2D('_1', '_2', xbins=100, ybins=100,
                                  scheduler=s)  # columns are called 1..30
        histogram2d.input.data = stirrer.output.table
        histogram2d.input['table', ('min', '_1', '_2')] = min_.output.table
        histogram2d.input['table', ('max', '_1', '_2')] = max_.output.table
        heatmap = Heatmap(filename='histo_%03d.png', scheduler=s)
        heatmap.input.array = histogram2d.output.table
        pr = Every(proc=self.terse, scheduler=s)
        pr.input.df = stirrer.output.table
        aio.run(s.start())
        last = histogram2d._table.last().to_dict()
        h1 = last['array']
        bounds =  [[last['ymin'], last['ymax']], [last['xmin'], last['xmax']]]
        t = stirrer._table.loc[:, ['_1', '_2']]
        v = t.to_array()
        bins = [histogram2d.params.ybins, histogram2d.params.xbins]
        h2 = fh.histogram2d(v[:,1], v[:,0], bins=bins, range=bounds)
        h2 = np.flip(h2, axis=0)
        self.assertEqual(np.sum(h1), np.sum(h2))
        self.assertListEqual(h1.reshape(-1).tolist(), h2.reshape(-1).tolist())

    def test_histogram2d4(self):
        s = self.scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        stirrer = StirrerView(update_column='_2',
                              fixed_step_size=1000, scheduler=s, delete_rows=5)
        stirrer.input.table = random.output.table
        min_ = Min(scheduler=s)
        min_.input.table = stirrer.output.table
        max_ = Max(scheduler=s)
        max_.input.table = stirrer.output.table
        histogram2d = MCHistogram2D('_1', '_2', xbins=100, ybins=100,
                                  scheduler=s)  # columns are called 1..30
        histogram2d.input.data = stirrer.output.table
        histogram2d.input['table', ('min', '_1', '_2')] = min_.output.table
        histogram2d.input['table', ('max', '_1', '_2')] = max_.output.table
        heatmap = Heatmap(filename='histo_%03d.png', scheduler=s)
        heatmap.input.array = histogram2d.output.table
        pr = Every(proc=self.terse, scheduler=s)
        pr.input.df = stirrer.output.table
        aio.run(s.start())
        last = histogram2d._table.last().to_dict()
        h1 = last['array']
        bounds =  [[last['ymin'], last['ymax']], [last['xmin'], last['xmax']]]
        v = stirrer._table.loc[:, ['_1', '_2']].to_array()
        bins = [histogram2d.params.ybins, histogram2d.params.xbins]
        h2 = fh.histogram2d(v[:,1], v[:,0], bins=bins, range=bounds)
        h2 = np.flip(h2, axis=0)
        self.assertEqual(np.sum(h1), np.sum(h2))
        self.assertListEqual(h1.reshape(-1).tolist(), h2.reshape(-1).tolist())        

    def test_histogram2d2(self):
        return self.t_histogram2d_impl(delete_rows=5)

    def test_histogram2d3(self):
        return self.t_histogram2d_impl(update_rows=5)

        
if __name__ == '__main__':
    ProgressiveTest.main()
