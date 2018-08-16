from . import ProgressiveTest, skip

import time

from progressivis import Print, Every
from progressivis.io import CSVLoader
from progressivis.table.select import Select
from progressivis.stats import Sample
from progressivis.datasets import get_dataset

from progressivis.core.bitmap import bitmap

import pandas as pd


def print_repr(x):
    print(repr(x))

class TestSelect(ProgressiveTest):
    def setUp(self):
        super(TestSelect, self).setUp()
#        self.log(self.INFO)

    def test_select_simple(self):
        s = self.scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,scheduler=s)
        sample=Sample(samples=100, scheduler=s)
        sample.input.table = csv.output.table
        q=Select(scheduler=s)
        q.input.table = csv.output.table
        q.input.select = sample.output.select
        prlen = Print(proc=self.terse,  scheduler=s)
        prlen.input.df = q.output.table
        s.start()
        s.join()
        print(repr(q.table()))
        self.assertEqual(len(q.table()), 100)
        self.assertEqual(bitmap(q.table().index), sample.get_data("select"))

    @skip("Need to implement select on tables")
    def test_select(self):
        s = self.scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False,header=None,force_valid_ids=True,scheduler=s)
        cst=Constant(pd.DataFrame({'query': ['_1 < 0.5']}),scheduler=s)
        q=Select(scheduler=s)
        q.input.df = csv.output.df
        q.input.query = cst.output.df
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input.df = q.output.df
        s.start()
        s.join()
        self.assertTrue(len(q.table()) < 1000000)


if __name__ == '__main__':
    ProgressiveTest.main()
