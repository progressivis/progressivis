from __future__ import annotations

from . import ProgressiveTest, skip

from progressivis import Print, Every
from progressivis.io import CSVLoader
from progressivis.utils import PDict
from progressivis.table.select import Select
from progressivis.table.constant import Constant
from progressivis.stats import Sample
from progressivis.datasets import get_dataset
from progressivis.core import aio
from progressivis.core.pintset import PIntSet

from typing import Any


def print_repr(x: Any) -> None:
    print(repr(x))


class TestSelect(ProgressiveTest):
    @skip("Too long")
    def test_select_simple(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        sample = Sample(samples=100, scheduler=s)
        sample.input[0] = csv.output.result
        q = Select(scheduler=s)
        q.input[0] = csv.output.result
        q.input.select = sample.output.select
        prlen = Print(proc=self.terse, scheduler=s)
        prlen.input[0] = q.output.result
        aio.run(s.start())
        print(repr(q.table))
        self.assertEqual(len(q.table), 100)
        self.assertEqual(PIntSet(q.table.index), sample.get_data("select"))

    @skip("Need to implement select on tables")
    def test_select(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"),
            index_col=False,
            header=None,
            force_valid_ids=True,
            scheduler=s,
        )
        cst = Constant(PDict({"query": ["_1 < 0.5"]}), scheduler=s)
        q = Select(scheduler=s)
        q.input[0] = csv.output.df
        q.input.query = cst.output.df
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = q.output.df
        aio.run(s.start())
        self.assertTrue(len(q.table) < 1000000)


if __name__ == "__main__":
    ProgressiveTest.main()
