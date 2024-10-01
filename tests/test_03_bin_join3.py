from . import ProgressiveTest, skip

from progressivis import Print, Every, CSVLoader, Constant, PTable
from progressivis.stats.api import Stats
from progressivis.datasets import get_dataset
from progressivis.table.bin_join import BinJoin
from progressivis.table.reduce import Reduce
from progressivis.core import aio
import pandas as pd

from typing import Any


def print_len(x: Any) -> None:
    if x is not None:
        print(len(x))


class TestJoin3(ProgressiveTest):
    @skip("Need fixing")
    def test_join(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        stat1 = Stats(1, reset_index=True, scheduler=s)
        stat1.input[0] = csv.output.result
        stat2 = Stats(2, reset_index=True, scheduler=s)
        stat2.input[0] = csv.output.result
        stat3 = Stats(3, reset_index=True, scheduler=s)
        stat3.input[0] = csv.output.result
        join = Reduce.expand(
            BinJoin,
            "first",
            "second",
            "result",
            [stat1.output.stats, stat2.output.stats, stat3.output.stats],
            scheduler=s,
        )
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = join.output.result
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = csv.output.result
        aio.run(s.start())
        res = join.trace_stats(max_runs=1)
        print(res)

    def test_join_simple(self) -> None:
        s = self.scheduler()
        cst1 = Constant(
            PTable(
                name="test_join_simple_cst1",
                data=pd.DataFrame({"xmin": [1], "xmax": [2]}),
                create=True,
            ),
            scheduler=s,
        )
        cst2 = Constant(
            PTable(
                name="test_join_simple_cst2",
                data=pd.DataFrame({"ymin": [3], "ymax": [4]}),
                create=True,
            ),
            scheduler=s,
        )
        cst3 = Constant(
            PTable(
                name="test_join_simple_cst3",
                data=pd.DataFrame({"zmin": [5], "zmax": [6]}),
                create=True,
            ),
            scheduler=s,
        )
        join = Reduce.expand(
            BinJoin,
            "first",
            "second",
            "result",
            [cst1.output.result, cst2.output.result, cst3.output.result],
            scheduler=s,
        )
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = join.output.result
        aio.run(s.start())
        res = join.trace_stats(max_runs=1)
        print(res)
        assert isinstance(join, BinJoin)
        df = join.result
        assert df is not None
        last = df.loc[df.index[-1]]
        assert last is not None
        self.assertTrue(
            last["xmin"] == 1
            and last["xmax"] == 2
            and last["ymin"] == 3
            and last["ymax"] == 4
            and last["zmin"] == 5
            and last["zmax"] == 6
        )


if __name__ == "__main__":
    ProgressiveTest.main()
