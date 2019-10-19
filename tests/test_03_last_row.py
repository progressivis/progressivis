from . import ProgressiveTest

from progressivis import Print, Every
from progressivis.table.last_row import LastRow
from progressivis.table.constant import Constant
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset
from progressivis.table.table import Table
from progressivis.table.join import Join
from progressivis.core.utils import get_random_name
import pandas as pd
import asyncio as aio

def print_len(x):
    if x is not None:
        print(len(x))


class TestLastRow(ProgressiveTest):
    def test_last_row(self):
        s = self.scheduler()
        csv = CSVLoader(get_dataset('smallfile'), index_col=False,header=None,scheduler=s)
        lr1 = LastRow(scheduler=s)
        lr1.input.table = csv.output.table
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input.df = lr1.output.table
        aio.run(s.start())
        df = csv.table()
        last = df.last()
        res = lr1.table()
        self.assertEqual(res.at[0,'_1'], last['_1'])

        #print(res)
    def test_last_row_simple(self):
        s = self.scheduler()
        t1 = Table(name=get_random_name("cst1"), data={'xmin': [1], 'xmax': [2]})
        t2 = Table(name=get_random_name("cst2"),data={'ymin': [3], 'ymax': [4]})
        cst1=Constant(t1, scheduler=s)
        cst2=Constant(t2, scheduler=s)
        join=Join(scheduler=s)
        join.input.table = cst1.output.table
        join.input.table = cst2.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = join.output.table
        aio.run(s.start())
        #res = join.trace_stats(max_runs=1)
        #pd.set_option('display.expand_frame_repr', False)
        #print(res)
        df = join.table()
        last = df.last()
        self.assertTrue(last['xmin']==1 and last['xmax']==2 and \
                        last['ymin']==3 and last['ymax']==4)


if __name__ == '__main__':
    ProgressiveTest.main()
