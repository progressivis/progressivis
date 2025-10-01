from . import ProgressiveTest, skip

from progressivis import Every, Print, Stats, CSVLoader, Merge, Constant, PTable, get_dataset
from progressivis.core import aio
import pandas as pd

# from pprint import pprint


@skip("Not ready")  # TODO: FIX IT
class TestMerge(ProgressiveTest):
    def test_merge(self) -> None:
        s = self.scheduler
        csv = CSVLoader(
            get_dataset("bigfile"), header=None, scheduler=s
        )
        stat1 = Stats(1, scheduler=s)
        stat1.input[0] = csv.output.result
        stat2 = Stats(2, scheduler=s)
        stat2.input[0] = csv.output.result
        merge = Merge(left_index=True, right_index=True, scheduler=s)
        merge.input[0] = stat1.output.result
        merge.input[0] = stat2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = merge.output.result
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = csv.output.result
        aio.run(s.start())
        _ = merge.trace_stats(max_runs=1)

    def test_merge_simple(self) -> None:
        s = self.scheduler
        cst1 = Constant(
            PTable(name=None, data=pd.DataFrame({"xmin": [1], "xmax": [2]})), scheduler=s
        )
        cst2 = Constant(
            PTable(name=None, data=pd.DataFrame({"ymin": [3], "ymax": [4]})), scheduler=s
        )
        merge = Merge(left_index=True, right_index=True, scheduler=s)
        merge.input[0] = cst1.output.result
        merge.input[0] = cst2.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = merge.output.result
        aio.run(s.start())
        _ = merge.trace_stats(max_runs=1)
        df = merge.result
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
