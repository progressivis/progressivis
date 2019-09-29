from . import ProgressiveTest, skip

from progressivis import Print, Every
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset
from progressivis.table.bin_join import BinJoin
from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis.table.reduce import Reduce

import pandas as pd


def print_len(x):
    if x is not None:
        print(len(x))


class TestJoin3(ProgressiveTest):
    @skip("Need fixing")
    def test_join(self):
        s=self.scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,scheduler=s)
        stat1=Stats(1, reset_index=True, scheduler=s)
        stat1.input.table = csv.output.table
        stat2=Stats(2, reset_index=True, scheduler=s)
        stat2.input.table = csv.output.table
        stat3=Stats(3, reset_index=True, scheduler=s)
        stat3.input.table = csv.output.table
        #join=Join(scheduler=s)
        #import pdb;pdb.set_trace()
        reduce_ = Reduce(BinJoin, "first", "second", "table", scheduler=s)
        reduce_.input.table = stat1.output.stats
        reduce_.input.table = stat2.output.stats
        join = reduce_.expand()
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = join.output.table
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input.df = csv.output.table
        s.start()
        res = join.trace_stats(max_runs=1)
        print(res)

    def test_join_simple(self):
        s=self.scheduler()
        cst1 = Constant(Table(name='test_join_simple_cst1',
                              data=pd.DataFrame({'xmin': [1], 'xmax': [2]}),
                              create=True), scheduler=s)
        cst2 = Constant(Table(name='test_join_simple_cst2',
                              data=pd.DataFrame({'ymin': [3], 'ymax': [4]}),
                              create=True), scheduler=s)
        cst3 = Constant(Table(name='test_join_simple_cst3',
                              data=pd.DataFrame({'zmin': [5], 'zmax': [6]}),
                              create=True), scheduler=s)
        # join=Join(scheduler=s)
        # reduce_ = Reduce(BinJoin, "first", "second", "table", scheduler=s)
        # reduce_.input.table = cst1.output.table
        # reduce_.input.table = cst2.output.table
        # reduce_.input.table = cst3.output.table
        # join = reduce_.expand()
        join = Reduce.expand(BinJoin, "first", "second", "table",
                             [cst1.output.table,
                              cst2.output.table,
                              cst3.output.table],
                             scheduler=s)
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = join.output.table
        s.start()
        s.join()
        res = join.trace_stats(max_runs=1)
        print(res)
        df = join.table()
        last = df.loc[df.index[-1]]
        self.assertTrue(last['xmin'] == 1 and last['xmax'] == 2 and
                        last['ymin'] == 3 and last['ymax'] == 4 and
                        last['zmin'] == 5 and last['zmax'] == 6)


if __name__ == '__main__':
    ProgressiveTest.main()
