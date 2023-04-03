"Test for Range Query"
from __future__ import annotations

from progressivis.table.constant import ConstDict
from progressivis import Print, Scheduler
from progressivis.stats import RandomPTable
from progressivis.core.pintset import PIntSet
from progressivis.table.range_query_2d import RangeQuery2d
from progressivis.utils.psdict import PDict
from progressivis.core import aio
from . import ProgressiveTest

from typing import cast


class TestRangeQuery(ProgressiveTest):
    "Test Suite for RangeQuery Module"

    def tearDown(self) -> None:
        TestRangeQuery.cleanup()

    def _range_query_impl(self, lo_x: float, lo_y: float, up_x: float, up_y: float) -> None:
        "Run tests of the RangeQuery module"
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=1000, scheduler=s)
            t_min = PDict({"_1": lo_x, "_2": lo_y})
            min_value = ConstDict(pdict=t_min, scheduler=s)
            t_max = PDict({"_1": up_x, "_2": up_y})
            max_value = ConstDict(pdict=t_max, scheduler=s)
            range_qry = RangeQuery2d(column_x="_1", column_y="_2", scheduler=s)
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
            .eval(f"(_1>{lo_x})&(_1<{up_x})&(_2>{lo_y})&(_2<{up_y})", result_object="index")
        )
        assert range_qry.result is not None
        self.assertEqual(range_qry.result.index, PIntSet(idx))

    def test_range_query_03_04_06_07(self) -> None:
        "Run tests of the RangeQuery module"
        self._range_query_impl(0.3, 0.4, 0.6, 0.7)

    def test_range_query_02_03_08_09(self) -> None:
        "Run tests of the RangeQuery module"
        self._range_query_impl(0.2, 0.3, 0.8, 0.9)

    def test_range_query_01_02_08_09(self) -> None:
        "Run tests of the RangeQuery module"
        self._range_query_impl(0.1, 0.2, 0.8, 0.9)

    def test_hist_index_min_max(self) -> None:
        "Test min_out and max_out on HistogramIndex"
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=100000, scheduler=s)
            t_min = PDict({"_1": 0.3, "_2": 0.4})
            min_value = ConstDict(pdict=t_min, scheduler=s)
            t_max = PDict({"_1": 0.8, "_2": 0.9})
            max_value = ConstDict(pdict=t_max, scheduler=s)
            range_qry = RangeQuery2d(column_x="_1", column_y="_2", scheduler=s)
            range_qry.create_dependent_modules(
                random, "result", min_value=min_value, max_value=max_value
            )
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = range_qry.output.result
            hist_index_x = range_qry.dep.hist_index_x
            assert hist_index_x is not None
            prt2_x = Print(proc=self.terse, scheduler=s)
            prt2_x.input[0] = hist_index_x.output.min_out
            pr3_x = Print(proc=self.terse, scheduler=s)
            pr3_x.input[0] = hist_index_x.output.max_out
            hist_index_y = range_qry.dep.hist_index_y
            assert hist_index_y is not None
            prt2_y = Print(proc=self.terse, scheduler=s)
            prt2_y.input[0] = hist_index_y.output.min_out
            pr3_y = Print(proc=self.terse, scheduler=s)
            pr3_y.input[0] = hist_index_y.output.max_out
        aio.run(s.start())
        assert random.result is not None
        res1 = cast(float, random.result.min()["_1"])
        assert hist_index_x.min_out is not None
        res2 = cast(float, hist_index_x.min_out["_1"])
        self.assertAlmostEqual(res1, res2)
        res1 = cast(float, random.result.max()["_1"])
        assert hist_index_x.max_out is not None
        res2 = cast(float, hist_index_x.max_out["_1"])
        self.assertAlmostEqual(res1, res2)
        assert random.result is not None
        res1 = cast(float, random.result.min()["_2"])
        assert hist_index_y.min_out is not None
        res2 = cast(float, hist_index_y.min_out["_2"])
        self.assertAlmostEqual(res1, res2)
        res1 = cast(float, random.result.max()["_2"])
        assert hist_index_y.max_out is not None
        res2 = cast(float, hist_index_y.max_out["_2"])
        self.assertAlmostEqual(res1, res2)

    def _query_min_max_impl(
        self, random: RandomPTable, t_min: PDict, t_max: PDict, s: Scheduler
    ) -> RangeQuery2d:
        min_value = ConstDict(pdict=t_min, scheduler=s)
        max_value = ConstDict(pdict=t_max, scheduler=s)
        range_qry = RangeQuery2d(column_x="_1", column_y="_2", scheduler=s)
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
            random = RandomPTable(2, rows=100000, scheduler=s)
            t_min = PDict({"_1": 0.3, "_2": 0.4})
            t_max = PDict({"_1": 0.8, "_2": 0.9})
            range_qry = self._query_min_max_impl(random, t_min, t_max, s)
        aio.run(s.start())
        min_data = range_qry.output.min.data()
        max_data = range_qry.output.max.data()
        self.assertAlmostEqual(min_data["_1"], 0.3)
        self.assertAlmostEqual(max_data["_1"], 0.8)
        self.assertAlmostEqual(min_data["_2"], 0.4)
        self.assertAlmostEqual(max_data["_2"], 0.9)

    def test_range_query_min_max2(self) -> None:
        "Test min and max on RangeQuery output"
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=100000, scheduler=s)
            t_min = PDict({"_1": 0.0, "_2": 0.1})
            t_max = PDict({"_1": float("nan"), "_2": 0.9})
            range_qry = self._query_min_max_impl(random, t_min, t_max, s)
        aio.run(s.start())
        min_data = cast(PDict, range_qry.output.min.data())
        max_data = range_qry.output.max.data()
        assert random.result is not None
        min_rand = random.result.min()["_1"]
        self.assertAlmostEqual(min_data["_1"], min_rand, delta=0.0001)
        self.assertAlmostEqual(max_data["_1"], 1.0, delta=0.0001)
        self.assertAlmostEqual(min_data["_2"], 0.1)
        self.assertAlmostEqual(max_data["_2"], 0.9)

    def test_range_query_min_max3(self) -> None:
        "Test min and max on RangeQuery output"
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=100000, scheduler=s)
            t_min = PDict({"_1": 0.3, "_2": 0.4})
            t_max = PDict({"_1": 15000.0, "_2": 500})
            range_qry = self._query_min_max_impl(random, t_min, t_max, s)
        aio.run(s.start())
        min_data = cast(PDict, range_qry.output.min.data())
        max_data = cast(PDict, range_qry.output.max.data())
        assert random.result is not None
        self.assertAlmostEqual(min_data["_1"], 0.3)
        max_rand_x = random.result.max()["_1"]
        self.assertAlmostEqual(max_data["_1"], max_rand_x)
        max_rand_y = random.result.max()["_2"]
        self.assertAlmostEqual(min_data["_2"], 0.4)
        self.assertAlmostEqual(max_data["_2"], max_rand_y)


if __name__ == "__main__":
    ProgressiveTest.main()
