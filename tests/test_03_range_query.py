"Test for Range Query"
from __future__ import annotations

from progressivis.table.constant import Constant
from progressivis import Print, Scheduler
from progressivis.stats import RandomTable, Min, Max
from progressivis.core.bitmap import bitmap
from progressivis.table.range_query import RangeQuery
from progressivis.utils.psdict import PsDict
from progressivis.core import aio
from . import ProgressiveTest, main

from typing import cast


class TestRangeQuery(ProgressiveTest):
    "Test Suite for RangeQuery Module"

    def tearDown(self) -> None:
        TestRangeQuery.cleanup()

    def test_range_query(self) -> None:
        "Run tests of the RangeQuery module"
        s = self.scheduler()
        with s:
            random = RandomTable(2, rows=1000, scheduler=s)
            t_min = PsDict({"_1": 0.3})
            min_value = Constant(table=t_min, scheduler=s)
            t_max = PsDict({"_1": 0.8})
            max_value = Constant(table=t_max, scheduler=s)
            range_qry = RangeQuery(column="_1", scheduler=s)
            range_qry.create_dependent_modules(
                random, "result", min_value=min_value, max_value=max_value
            )
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = range_qry.output.result
        aio.run(s.start())
        assert range_qry.input_module is not None
        idx = (
            range_qry.input_module.output["result"]
            .data()
            .eval("(_1>0.3)&(_1<0.8)", result_object="index")
        )
        self.assertEqual(range_qry.table.index, bitmap(idx))

    def test_hist_index_min_max(self) -> None:
        "Test min_out and max_out on HistogramIndex"
        s = self.scheduler()
        with s:
            random = RandomTable(2, rows=100000, scheduler=s)
            t_min = PsDict({"_1": 0.3})
            min_value = Constant(table=t_min, scheduler=s)
            t_max = PsDict({"_1": 0.8})
            max_value = Constant(table=t_max, scheduler=s)
            range_qry = RangeQuery(column="_1", scheduler=s)
            range_qry.create_dependent_modules(
                random, "result", min_value=min_value, max_value=max_value
            )
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = range_qry.output.result
            hist_index = range_qry.hist_index
            assert hist_index is not None
            min_ = Min(name="min_" + str(hash(hist_index)), scheduler=s)
            min_.input[0] = hist_index.output.min_out
            prt2 = Print(proc=self.terse, scheduler=s)
            prt2.input[0] = min_.output.result
            max_ = Max(name="max_" + str(hash(hist_index)), scheduler=s)
            max_.input[0] = hist_index.output.max_out
            pr3 = Print(proc=self.terse, scheduler=s)
            pr3.input[0] = max_.output.result
        aio.run(s.start())
        res1 = cast(float, random.table.min()["_1"])
        res2 = cast(float, min_.psdict["_1"])
        self.assertAlmostEqual(res1, res2)
        res1 = cast(float, random.table.max()["_1"])
        res2 = cast(float, max_.psdict["_1"])
        self.assertAlmostEqual(res1, res2)

    def _query_min_max_impl(
        self, random: RandomTable, t_min: PsDict, t_max: PsDict, s: Scheduler
    ) -> RangeQuery:
        min_value = Constant(table=t_min, scheduler=s)
        max_value = Constant(table=t_max, scheduler=s)
        range_qry = RangeQuery(column="_1", scheduler=s)
        range_qry.create_dependent_modules(
            random, "result", min_value=min_value, max_value=max_value
        )
        prt = Print(proc=self.terse, scheduler=s)
        prt.input[0] = range_qry.output.result
        prt2 = Print(proc=self.terse, scheduler=s)
        prt2.input[0] = range_qry.output.min
        pr3 = Print(proc=self.terse, scheduler=s)
        pr3.input[0] = range_qry.output.max
        return range_qry

    def test_range_query_min_max(self) -> None:
        "Test min and max on RangeQuery output"
        s = self.scheduler()
        with s:
            random = RandomTable(2, rows=100000, scheduler=s)
            t_min = PsDict({"_1": 0.3})
            t_max = PsDict({"_1": 0.8})
            range_qry = self._query_min_max_impl(random, t_min, t_max, s)
        aio.run(s.start())
        min_data = range_qry.output.min.data()
        max_data = range_qry.output.max.data()
        self.assertAlmostEqual(min_data["_1"], 0.3)
        self.assertAlmostEqual(max_data["_1"], 0.8)

    def test_range_query_min_max2(self) -> None:
        "Test min and max on RangeQuery output"
        s = self.scheduler()
        with s:
            random = RandomTable(2, rows=100000, scheduler=s)
            t_min = PsDict({"_1": 0.0})
            t_max = PsDict({"_1": float("nan")})
            range_qry = self._query_min_max_impl(random, t_min, t_max, s)
        aio.run(s.start())
        min_data = cast(PsDict, range_qry.output.min.data())
        max_data = range_qry.output.max.data()
        min_rand = random.table.min()["_1"]
        self.assertAlmostEqual(min_data["_1"], min_rand, delta=0.0001)
        self.assertAlmostEqual(max_data["_1"], 1.0, delta=0.0001)

    def test_range_query_min_max3(self) -> None:
        "Test min and max on RangeQuery output"
        s = self.scheduler()
        with s:
            random = RandomTable(2, rows=100000, scheduler=s)
            t_min = PsDict({"_1": 0.3})
            t_max = PsDict({"_1": 15000.0})
            range_qry = self._query_min_max_impl(random, t_min, t_max, s)
        aio.run(s.start())
        min_data = cast(PsDict, range_qry.output.min.data())
        max_data = cast(PsDict, range_qry.output.max.data())
        max_rand = random.table.max()["_1"]
        self.assertAlmostEqual(min_data["_1"], 0.3)
        self.assertAlmostEqual(max_data["_1"], max_rand)


if __name__ == "__main__":
    main()
