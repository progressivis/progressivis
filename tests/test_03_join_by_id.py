from . import ProgressiveTest

from progressivis import Print, Every
from progressivis.stats import Stats
from progressivis.io import CSVLoader
from progressivis.datasets import get_dataset
from progressivis.table.join_by_id import Join
from progressivis.table.constant import Constant
from progressivis.table.table import PTable
from progressivis.core import aio
import pandas as pd

from typing import Any

# from pprint import pprint


def print_len(x: Any) -> None:
    if x is not None:
        print(len(x))


class TestJoin(ProgressiveTest):
    def test_join(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        stat1 = Stats(1, reset_index=True, scheduler=s)
        stat1.input[0] = csv.output.result
        stat2 = Stats(2, reset_index=True, scheduler=s)
        stat2.input[0] = csv.output.result
        join = Join(scheduler=s)
        join.input[0] = stat1.output.result
        join.input[0] = stat2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = join.output.result
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = csv.output.result
        aio.run(s.start())
        res = join.trace_stats(max_runs=1)
        print(res)

    def te_st_join_simple(self) -> None:
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
        join = Join(scheduler=s)
        join.input[0] = cst1.output.result
        join.input[0] = cst2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = join.output.result
        aio.run(s.start())
        res = join.trace_stats(max_runs=1)
        print(res)
        df = join.table
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
