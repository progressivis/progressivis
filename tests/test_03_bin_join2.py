import pandas as pd

from progressivis.core import aio
from progressivis import Tick, CSVLoader, Constant, PTable
from progressivis.stats.api import Stats
from progressivis.datasets import get_dataset
from progressivis.table.bin_join import BinJoin
from progressivis.table.reduce import Reduce

from . import ProgressiveTest, skip


class TestJoin2(ProgressiveTest):
    @skip("Need fixing")
    def test_join(self) -> None:
        s = self.scheduler
        csv = CSVLoader(
            get_dataset("bigfile"), header=None, scheduler=s
        )
        stat1 = Stats(1, reset_index=True, scheduler=s)
        stat1.input[0] = csv.output.result
        stat2 = Stats(2, reset_index=True, scheduler=s)
        stat2.input[0] = csv.output.result
        # join=Join(scheduler=s)
        # reduce_ = Reduce(BinJoin, "first", "second", "table", scheduler=s)
        # reduce_.input[0] = stat1.output.stats
        # reduce_.input[0] = stat2.output.stats
        # join = reduce_.expand()
        join = Reduce.expand(
            BinJoin,
            "first",
            "second",
            "table",
            [stat1.output.stats, stat2.output.stats],
            scheduler=s,
        )
        pr = Tick(scheduler=s)
        pr.input[0] = join.output.result
        prlen = Tick(scheduler=s)
        prlen.input[0] = csv.output.result
        aio.run(s.start())
        res = join.trace_stats(max_runs=1)
        print(res)

    def test_join_simple(self) -> None:
        s = self.scheduler
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
        join = Reduce.expand(
            BinJoin,
            "first",
            "second",
            "table",
            [cst1.output.result, cst2.output.result],
            scheduler=s,
        )
        pr = Tick(scheduler=s)
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
        )


if __name__ == "__main__":
    ProgressiveTest.main()
