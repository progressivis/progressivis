from . import ProgressiveTest
from progressivis.core import aio
from progressivis import Scheduler, Every
from progressivis.io import CSVLoader
from progressivis.stats import Histogram1D, Min, Max
from progressivis.datasets import get_dataset
from progressivis.table.stirrer import Stirrer
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)

class TestHistogram1D(ProgressiveTest):

    #def tearDown(self):
        #StorageManager.default.end()


    def test_histogram1d(self):
        s=self.scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,scheduler=s)
        min_ = Min(scheduler=s)
        min_.input.table = csv.output.table
        max_ = Max(scheduler=s)
        max_.input.table = csv.output.table
        histogram1d=Histogram1D('_2', scheduler=s) # columns are called 1..30
        histogram1d.input.table = csv.output.table
        histogram1d.input.min = min_.output.table
        histogram1d.input.max = max_.output.table
        pr = Every(proc=self.terse, scheduler=s)
        pr.input.df = csv.output.table
        aio.run(s.start())
        s = histogram1d.trace_stats()


    def test_histogram1d1(self):
        s=self.scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,scheduler=s)
        min_ = Min(scheduler=s)
        min_.input.table = csv.output.table
        max_ = Max(scheduler=s)
        max_.input.table = csv.output.table
        histogram1d=Histogram1D('_2', scheduler=s) # columns are called 1..30
        histogram1d.input.table = csv.output.table
        histogram1d.input.min = min_.output.table
        histogram1d.input.max = max_.output.table
        pr = Every(proc=self.terse, scheduler=s)
        pr.input.df = csv.output.table
        aio.run(s.start())
        s = histogram1d.trace_stats()
        last = histogram1d._table.last().to_dict()
        h1 = last['array']
        bounds = (last['min'], last['max'])
        df = pd.read_csv(get_dataset('bigfile'), header=None, usecols=[2])
        v = df.to_numpy().reshape(-1)
        h2, _ = np.histogram(v, bins=histogram1d.params.bins, density=False, range=bounds)
        self.assertListEqual(h1.tolist(), h2.tolist())

    def t_histogram1d_impl(self, **kw):
        s=self.scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,scheduler=s)
        stirrer = Stirrer(update_column='_2',
                          fixed_step_size=1000, scheduler=s, **kw)
        stirrer.input.table = csv.output.table
        min_ = Min(scheduler=s)
        min_.input.table = stirrer.output.table
        max_ = Max(scheduler=s)
        max_.input.table = stirrer.output.table
        histogram1d=Histogram1D('_2', scheduler=s) # columns are called 1..30
        histogram1d.input.table = stirrer.output.table
        histogram1d.input.min = min_.output.table
        histogram1d.input.max = max_.output.table
   
        #pr = Print(scheduler=s)
        pr = Every(proc=self.terse, scheduler=s)
        pr.input.df = stirrer.output.table
        aio.run(s.start())
        s = histogram1d.trace_stats()
        last = histogram1d._table.last().to_dict()
        h1 = last['array']
        bounds = (last['min'], last['max'])
        v = stirrer._table.loc[:, ['_2']].to_array().reshape(-1)
        h2, _ = np.histogram(v, bins=histogram1d.params.bins, density=False, range=bounds)
        self.assertEqual(np.sum(h1), np.sum(h2))
        self.assertTrue(np.allclose(h1, h2, atol=1.0))

    def test_histogram1d2(self):
        return self.t_histogram1d_impl(delete_rows=5)

    def test_histogram1d3(self):
        return self.t_histogram1d_impl(update_rows=5)

if __name__ == '__main__':
    ProgressiveTest.main()
