from progressivis.table.table import PTable
from progressivis.table.constant import Constant
from progressivis import Print
from progressivis.stats import RandomPTable
from progressivis.table.bisectmod import Bisect
from progressivis.core.pintset import PIntSet
from progressivis.table.hist_index import HistogramIndex
from progressivis.core import aio
from progressivis.table.stirrer import Stirrer
from . import ProgressiveTest


class TestBisect(ProgressiveTest):
    def test_bisect(self) -> None:
        s = self.scheduler()
        random = RandomPTable(2, rows=1000_000, scheduler=s)
        t = PTable(name=None, dshape="{value: string}", data={"value": [0.5]})
        min_value = Constant(table=t, scheduler=s)
        hist_index = HistogramIndex(column="_1", scheduler=s)
        hist_index.create_dependent_modules(random, "result")
        bisect_ = Bisect(column="_1", op=">", hist_index=hist_index, scheduler=s)
        bisect_.input[0] = hist_index.output.result
        # bisect_.input[0] = random.output.result
        bisect_.input.limit = min_value.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = bisect_.output.result
        aio.run(s.start())
        assert random.result is not None
        assert bisect_.result is not None
        idx = random.result.eval("_1>0.5", result_object="index")
        self.assertEqual(bisect_.result.index, PIntSet(idx))

    def test_bisect2(self) -> None:
        s = self.scheduler()
        random = RandomPTable(2, rows=100_000, scheduler=s)
        stirrer = Stirrer(
            update_column="_1",
            delete_rows=100,
            # update_rows=5,
            # fixed_step_size=100,
            scheduler=s,
        )
        stirrer.input[0] = random.output.result
        t = PTable(name=None, dshape="{value: string}", data={"value": [0.5]})
        min_value = Constant(table=t, scheduler=s)
        hist_index = HistogramIndex(column="_1", scheduler=s)
        hist_index.create_dependent_modules(stirrer, "result")
        bisect_ = Bisect(column="_1", op=">", hist_index=hist_index, scheduler=s)
        bisect_.input[0] = hist_index.output.result
        # bisect_.input[0] = random.output.result
        bisect_.input.limit = min_value.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = bisect_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert bisect_.result is not None
        idx = stirrer.result.eval("_1>0.5", result_object="index")
        self.assertEqual(bisect_.result.index, PIntSet(idx))

    def test_bisect3(self) -> None:
        s = self.scheduler()
        random = RandomPTable(2, rows=100_000, scheduler=s)
        stirrer = Stirrer(
            update_column="_1", update_rows=100, fixed_step_size=100, scheduler=s
        )
        stirrer.input[0] = random.output.result
        t = PTable(name=None, dshape="{value: string}", data={"value": [0.5]})
        min_value = Constant(table=t, scheduler=s)
        hist_index = HistogramIndex(column="_1", scheduler=s)
        hist_index.create_dependent_modules(stirrer, "result")
        bisect_ = Bisect(column="_1", op=">", hist_index=hist_index, scheduler=s)
        bisect_.input[0] = hist_index.output.result
        # bisect_.input[0] = random.output.result
        bisect_.input.limit = min_value.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = bisect_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert bisect_.result is not None
        idx = stirrer.result.eval("_1>0.5", result_object="index")
        self.assertEqual(bisect_.result.index, PIntSet(idx))


if __name__ == "__main__":
    ProgressiveTest.main()
