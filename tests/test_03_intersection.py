from progressivis.table.table import PTable
from progressivis.table.constant import Constant
from progressivis import Print
from progressivis.stats import RandomPTable
from progressivis.table.bisectmod import Bisect
from progressivis.core.pintset import PIntSet
from progressivis.table.hist_index import HistogramIndex
from progressivis.table.intersection import Intersection
from progressivis.table.stirrer import Stirrer
from progressivis.core import aio

from . import ProgressiveTest

from typing import Any


class TestIntersection(ProgressiveTest):
    def test_intersection(self) -> None:
        s = self.scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        t_min = PTable(name=None, dshape="{_1: float64}", data={"_1": [0.3]})
        min_value = Constant(table=t_min, scheduler=s)
        t_max = PTable(name=None, dshape="{_1: float64}", data={"_1": [0.8]})
        max_value = Constant(table=t_max, scheduler=s)
        hist_index = HistogramIndex(column="_1", scheduler=s)
        hist_index.create_dependent_modules(random, "result")
        bisect_min = Bisect(column="_1", op=">", hist_index=hist_index, scheduler=s)
        bisect_min.input[0] = hist_index.output.result
        # bisect_.input[0] = random.output.result
        bisect_min.input.limit = min_value.output.result
        bisect_max = Bisect(column="_1", op="<", hist_index=hist_index, scheduler=s)
        bisect_max.input[0] = hist_index.output.result
        # bisect_.input[0] = random.output.result
        bisect_max.input.limit = max_value.output.result
        inter = Intersection(scheduler=s)
        inter.input[0] = bisect_min.output.result
        inter.input[0] = bisect_max.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = inter.output.result
        aio.run(s.start())
        assert hist_index.input_module is not None
        idx = (
            hist_index.input_module.output["result"]
            .data()
            .eval("(_1>0.3)&(_1<0.8)", result_object="index")
        )
        self.assertEqual(inter.table.index, PIntSet(idx))

    def _impl_stirred_tst_intersection(self, **kw: Any) -> None:
        s = self.scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(update_column="_2", fixed_step_size=1000, scheduler=s, **kw)
        stirrer.input[0] = random.output.result
        t_min = PTable(name=None, dshape="{_1: float64}", data={"_1": [0.3]})
        min_value = Constant(table=t_min, scheduler=s)
        t_max = PTable(name=None, dshape="{_1: float64}", data={"_1": [0.8]})
        max_value = Constant(table=t_max, scheduler=s)
        hist_index = HistogramIndex(column="_1", scheduler=s)
        hist_index.create_dependent_modules(stirrer, "result")
        bisect_min = Bisect(column="_1", op=">", hist_index=hist_index, scheduler=s)
        bisect_min.input[0] = hist_index.output.result
        # bisect_.input[0] = random.output.result
        bisect_min.input.limit = min_value.output.result
        bisect_max = Bisect(column="_1", op="<", hist_index=hist_index, scheduler=s)
        bisect_max.input[0] = hist_index.output.result
        # bisect_.input[0] = random.output.result
        bisect_max.input.limit = max_value.output.result
        inter = Intersection(scheduler=s)
        inter.input[0] = bisect_min.output.result
        inter.input[0] = bisect_max.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = inter.output.result
        aio.run(s.start())
        assert hist_index.input_module is not None
        idx = (
            hist_index.input_module.output["result"]
            .data()
            .eval("(_1>0.3)&(_1<0.8)", result_object="index")
        )
        self.assertEqual(inter.table.index, PIntSet(idx))

    def test_intersection2(self) -> None:
        self._impl_stirred_tst_intersection(delete_rows=5)

    def test_intersection3(self) -> None:
        self._impl_stirred_tst_intersection(update_rows=5)


if __name__ == "__main__":
    ProgressiveTest.main()
