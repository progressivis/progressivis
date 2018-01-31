from . import ProgressiveTest

from progressivis import Scheduler, Every
from progressivis.io import CSVLoader
from progressivis.stats import Histogram1D, Min, Max
from progressivis.datasets import get_dataset

#import pandas as pd

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
   
        #pr = Print(scheduler=s)
        pr = Every(proc=self.terse, scheduler=s)
        pr.input.df = csv.output.table
        s.start(tick_proc=lambda s,r: csv.is_terminated() and s.stop())
        s.join()
        s = histogram1d.trace_stats()
        #print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        #pd.set_option('display.expand_frame_repr', False)
        #print(repr(histogram1d.table()))

if __name__ == '__main__':
    ProgressiveTest.main()
