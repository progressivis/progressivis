from progressivis.table.constant import ConstDict
from progressivis import Print
from progressivis.stats import RandomPTable
from progressivis.table.simple_filter import SimpleFilter
from progressivis.core.pintset import PIntSet
from progressivis.utils.psdict import PDict
from progressivis.core import aio
from progressivis.table.stirrer import Stirrer
from . import ProgressiveTest


class TestSimpleFilter(ProgressiveTest):
    def test_filter(self) -> None:
        s = self.scheduler()
        random = RandomPTable(2, rows=100_000, scheduler=s)
        min_value = ConstDict(pdict=PDict({"value": 0.5}), scheduler=s)
        filter_ = SimpleFilter(column="_1", op=">", scheduler=s)
        filter_.create_dependent_modules(random, "result")
        filter_.input.value = min_value.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = filter_.output.result
        aio.run(s.start())
        assert random.result is not None
        assert filter_.result is not None
        idx = random.result.eval("_1>0.5", result_object="index")
        self.assertEqual(filter_.result.index, PIntSet(idx))

    def test_filter2(self) -> None:
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
        min_value = ConstDict(pdict=PDict({"value": 0.5}), scheduler=s)
        filter_ = SimpleFilter(column="_1", op=">", scheduler=s)
        filter_.create_dependent_modules(stirrer, "result")
        filter_.input.value = min_value.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = filter_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert filter_.result is not None
        idx = stirrer.result.eval("_1>0.5", result_object="index")
        self.assertEqual(filter_.result.index, PIntSet(idx))

    def test_filter3(self) -> None:
        s = self.scheduler()
        random = RandomPTable(2, rows=100_000, scheduler=s)
        stirrer = Stirrer(
            update_column="_1", update_rows=100, fixed_step_size=100, scheduler=s
        )
        stirrer.input[0] = random.output.result
        min_value = ConstDict(pdict=PDict({"value": 0.5}), scheduler=s)
        filter_ = SimpleFilter(column="_1", op=">", scheduler=s)
        filter_.create_dependent_modules(stirrer, "result")
        filter_.input.value = min_value.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = filter_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert filter_.result is not None
        idx = stirrer.result.eval("_1>0.5", result_object="index")
        self.assertEqual(filter_.result.index, PIntSet(idx))


if __name__ == "__main__":
    ProgressiveTest.main()
