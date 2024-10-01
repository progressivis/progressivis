from . import ProgressiveTest
from progressivis.core import aio
from progressivis import Print, Scheduler, RandomPTable, PIntSet
from progressivis.table.filtermod import FilterMod
from progressivis.table.stirrer import Stirrer
import pandas as pd


class TestFilter(ProgressiveTest):
    def test_filter(self) -> None:
        s = Scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        filter_ = FilterMod(expr="_1 > 0.5", scheduler=s)
        filter_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = filter_.output.result
        aio.run(s.start())
        idx = (
            filter_.get_input_slot("table").data().eval("_1>0.5", result_object="index")
        )
        assert filter_.result is not None
        self.assertEqual(filter_.result.index, PIntSet(idx))

    def test_filter_1(self) -> None:
        s = Scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        filter_ = FilterMod(scheduler=s)
        filter_.params.expr = "_1 > 0.5"
        filter_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = filter_.output.result
        aio.run(s.start())
        idx = (
            filter_.get_input_slot("table").data().eval("_1>0.5", result_object="index")
        )
        assert filter_.result is not None
        self.assertEqual(filter_.result.index, PIntSet(idx))

    def test_filter2(self) -> None:
        s = Scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(
            update_column="_1",
            delete_rows=5,
            # update_rows=5,
            fixed_step_size=100,
            scheduler=s,
        )
        stirrer.input[0] = random.output.result
        filter_ = FilterMod(expr="_1 > 0.5", scheduler=s)
        filter_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = filter_.output.result
        aio.run(s.start())
        assert filter_.result is not None
        tbl = filter_.get_input_slot("table").data()
        idx = tbl.eval("_1>0.5", result_object="index")
        self.assertEqual(filter_.result.index, PIntSet(idx))
        df = pd.DataFrame(tbl.to_dict(), index=tbl.index.to_array())
        dfe = df.eval("_1>0.5")
        self.assertEqual(filter_.result.index, PIntSet(df.index[dfe]))

    def test_filter3(self) -> None:
        s = Scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(
            update_column="_1", update_rows=5, fixed_step_size=100, scheduler=s
        )
        stirrer.input[0] = random.output.result
        filter_ = FilterMod(expr="_1 > 0.5", scheduler=s)
        filter_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = filter_.output.result
        aio.run(s.start())
        assert filter_.result is not None
        tbl = filter_.get_input_slot("table").data()
        idx = tbl.eval("_1>0.5", result_object="index")
        self.assertEqual(filter_.result.index, PIntSet(idx))
        df = pd.DataFrame(tbl.to_dict(), index=tbl.index.to_array())
        dfe = df.eval("_1>0.5")
        self.assertEqual(filter_.result.index, PIntSet(df.index[dfe]))


if __name__ == "__main__":
    ProgressiveTest.main()
