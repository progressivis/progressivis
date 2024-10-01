from progressivis.table.constant import ConstDict
from progressivis import Print, RandomPTable, PIntSet, PDict
from progressivis.table.simple_filter import SimpleFilter
from progressivis.table.intersection import Intersection
from progressivis.table.stirrer import Stirrer
from progressivis.core import aio

from . import ProgressiveTest

from typing import Any


class TestIntersection(ProgressiveTest):
    def test_intersection(self) -> None:
        s = self.scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        min_value = ConstDict(pdict=PDict({"_1": 0.3}), scheduler=s)
        max_value = ConstDict(pdict=PDict({"_1": 0.8}), scheduler=s)
        filter_min = SimpleFilter(column="_1", op=">", scheduler=s)
        filter_min.create_dependent_modules(random, "result")
        filter_min.input.value = min_value.output.result
        hist_index = filter_min.dep.hist_index  # sharing index between min and max
        filter_max = SimpleFilter(column="_1", op="<", scheduler=s)
        filter_max.create_dependent_modules(random, "result", hist_index=hist_index)
        filter_max.input.value = max_value.output.result
        inter = Intersection(scheduler=s)
        inter.input[0] = filter_min.output.result
        inter.input[0] = filter_max.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = inter.output.result
        aio.run(s.start())
        assert inter.result is not None
        idx = (
            random.output["result"]
            .data()
            .eval("(_1>0.3)&(_1<0.8)", result_object="index")
        )
        self.assertEqual(inter.result.index, PIntSet(idx))

    def _impl_stirred_tst_intersection(self, **kw: Any) -> None:
        s = self.scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(update_column="_2", fixed_step_size=1000, scheduler=s, **kw)
        stirrer.input[0] = random.output.result
        min_value = ConstDict(pdict=PDict({"_1": 0.3}), scheduler=s)
        max_value = ConstDict(pdict=PDict({"_1": 0.8}), scheduler=s)
        filter_min = SimpleFilter(column="_1", op=">", scheduler=s)
        filter_min.create_dependent_modules(stirrer, "result")
        filter_min.input.value = min_value.output.result
        hist_index = filter_min.dep.hist_index
        filter_max = SimpleFilter(column="_1", op="<", scheduler=s)
        filter_max.create_dependent_modules(stirrer, "result", hist_index=hist_index)
        filter_max.input.value = max_value.output.result
        inter = Intersection(scheduler=s)
        inter.input[0] = filter_min.output.result
        inter.input[0] = filter_max.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = inter.output.result
        aio.run(s.start())
        assert inter.result is not None
        idx = (
            stirrer.output["result"]
            .data()
            .eval("(_1>0.3)&(_1<0.8)", result_object="index")
        )
        self.assertEqual(inter.result.index, PIntSet(idx))

    def test_intersection2(self) -> None:
        self._impl_stirred_tst_intersection(delete_rows=5)

    def test_intersection3(self) -> None:
        self._impl_stirred_tst_intersection(update_rows=5)


if __name__ == "__main__":
    ProgressiveTest.main()
