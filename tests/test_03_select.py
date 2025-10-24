from __future__ import annotations

from . import ProgressiveTest, skip

from progressivis import Tick, CSVLoader, PDict, Select, ConstDict, Sample, get_dataset, PIntSet
from progressivis.core import aio

from typing import Any


def print_repr(x: Any) -> None:
    print(repr(x))


class TestSelect(ProgressiveTest):
    @skip("Too long")
    def test_select_simple(self) -> None:
        s = self.scheduler
        csv = CSVLoader(
            get_dataset("bigfile"), header=None, scheduler=s
        )
        sample = Sample(samples=100, scheduler=s)
        sample.input[0] = csv.output.result
        q = Select(scheduler=s)
        q.input[0] = csv.output.result
        q.input.select = sample.output.select
        prlen = Tick(scheduler=s)
        prlen.input[0] = q.output.result
        aio.run(s.start())
        assert q.result is not None
        print(repr(q.result))
        self.assertEqual(len(q.result), 100)
        self.assertEqual(PIntSet(q.result.index), sample.get_data("select"))

    @skip("Need to implement select on tables")
    def test_select(self) -> None:
        s = self.scheduler
        csv = CSVLoader(
            get_dataset("bigfile"),
            header=None,
            force_valid_ids=True,
            scheduler=s,
        )
        cst = ConstDict(PDict({"query": ["_1 < 0.5"]}), scheduler=s)
        q = Select(scheduler=s)
        q.input[0] = csv.output.df
        q.input.query = cst.output.df
        prlen = Tick(scheduler=s)
        prlen.input[0] = q.output.df
        aio.run(s.start())
        assert q.result is not None
        self.assertTrue(len(q.result) < 1000000)


if __name__ == "__main__":
    ProgressiveTest.main()
