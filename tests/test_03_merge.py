from . import ProgressiveTest

from progressivis import Every, Print
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.table.merge import Merge
from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis.datasets import get_dataset
from progressivis.core import aio
import pandas as pd
#from pprint import pprint


class TestMerge(ProgressiveTest):
    def test_merge(self):
        s = self.scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,scheduler=s)
        stat1=Stats(1, scheduler=s)
        stat1.input.table = csv.output.table
        stat2=Stats(2, scheduler=s)
        stat2.input.table = csv.output.table
        merge=Merge(left_index=True,right_index=True,scheduler=s)
        merge.input.table = stat1.output.table
        merge.input.table = stat2.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = merge.output.table
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input.df = csv.output.table
        aio.run(s.start())
        res = merge.trace_stats(max_runs=1)
        #pd.set_option('display.expand_frame_repr', False)
        #print(res)

    def test_merge_simple(self):
        s = self.scheduler()
        cst1=Constant(Table(name=None, data=pd.DataFrame({'xmin': [1], 'xmax': [2]})), scheduler=s)
        cst2=Constant(Table(name=None, data=pd.DataFrame({'ymin': [3], 'ymax': [4]})), scheduler=s)
        merge=Merge(left_index=True,right_index=True,scheduler=s)
        merge.input.table = cst1.output.table
        merge.input.table = cst2.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = merge.output.table
        aio.run(s.start())
        res = merge.trace_stats(max_runs=1)
        #pd.set_option('display.expand_frame_repr', False)
        #print(res)
        df = merge.table()
        last = df.loc[df.index[-1]]
        self.assertTrue(last['xmin']==1 and last['xmax']==2 and \
                        last['ymin']==3 and last['ymax']==4)

if __name__ == '__main__':
    ProgressiveTest.main()
