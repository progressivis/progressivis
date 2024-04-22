"Test for Range Query"
from __future__ import annotations

from progressivis.table.constant import ConstDict
from progressivis import Print, Scheduler
from progressivis.stats import RandomPTable
from progressivis.core.pintset import PIntSet
from progressivis.table.range_query import RangeQuery
from progressivis.utils.psdict import PDict
from progressivis.core import aio
from . import ProgressiveTest

from typing import cast


class TestRangeQuery(ProgressiveTest):
    "Test Suite for RangeQuery Module"

    def tearDown(self) -> None:
        TestRangeQuery.cleanup()

    def _range_query_impl(self, lo: float, up: float) -> None:
        "Run tests of the RangeQuery module"
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=20_000, scheduler=s)
            t_min = PDict({"_1": lo})
            min_value = ConstDict(pdict=t_min, scheduler=s)
            t_max = PDict({"_1": up})
            max_value = ConstDict(pdict=t_max, scheduler=s)
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
            .eval(f"(_1>{lo})&(_1<{up})", result_object="index")
        )
        assert range_qry.result is not None
        self.assertEqual(range_qry.result.index, PIntSet(idx))

    def _range_query_impl2(self, lo: float, up: float) -> None:
        "Run tests of the RangeQuery module"
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=20_000, scheduler=s)
            t_min_max = PDict({"lower": lo, "upper": up})
            min_max_value = ConstDict(pdict=t_min_max, scheduler=s)
            range_qry = RangeQuery(column="_1", scheduler=s)
            range_qry.params.watched_key_lower = "lower"
            range_qry.params.watched_key_upper = "upper"
            range_qry.create_dependent_modules(
                random, "result", min_value=min_max_value, max_value=min_max_value
            )
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = range_qry.output.result
        aio.run(s.start())
        assert range_qry.input_module is not None
        idx = (
            range_qry.input_module.output["result"]
            .data()
            .eval(f"(_1>{lo})&(_1<{up})", result_object="index")
        )
        assert range_qry.result is not None
        self.assertEqual(range_qry.result.index, PIntSet(idx))

    def test_range_query_04_06(self) -> None:
        "Run tests of the RangeQuery module"
        self._range_query_impl(0.4, 0.6)

    def test_range_query_04_06_bis(self) -> None:
        "Run tests of the RangeQuery module"
        self._range_query_impl2(0.4, 0.6)

    def test_range_query_03_08(self) -> None:
        "Run tests of the RangeQuery module"
        self._range_query_impl(0.3, 0.8)

    def test_range_query_01_09(self) -> None:
        "Run tests of the RangeQuery module"
        self._range_query_impl(0.1, 0.9)

    def test_hist_index_min_max(self) -> None:
        "Test min_out and max_out on HistogramIndex"
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=100_000, scheduler=s)
            t_min = PDict({"_1": 0.3})
            min_value = ConstDict(pdict=t_min, scheduler=s)
            t_max = PDict({"_1": 0.8})
            max_value = ConstDict(pdict=t_max, scheduler=s)
            range_qry = RangeQuery(column="_1", scheduler=s)
            range_qry.create_dependent_modules(
                random, "result", min_value=min_value, max_value=max_value
            )
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = range_qry.output.result
            hist_index = range_qry.dep.hist_index
            assert hist_index is not None
            prt2 = Print(proc=self.terse, scheduler=s)
            prt2.input[0] = hist_index.output.min_out
            pr3 = Print(proc=self.terse, scheduler=s)
            pr3.input[0] = hist_index.output.min_out
        aio.run(s.start())
        assert random.result is not None
        res1 = cast(float, random.result.min()["_1"])
        assert hist_index.min_out is not None
        res2 = cast(float, hist_index.min_out["_1"])
        self.assertAlmostEqual(res1, res2)
        res1 = cast(float, random.result.max()["_1"])
        assert hist_index.max_out is not None
        res2 = cast(float, hist_index.max_out["_1"])
        self.assertAlmostEqual(res1, res2)

    def _query_min_max_impl(
        self, random: RandomPTable, t_min: PDict, t_max: PDict, s: Scheduler
    ) -> RangeQuery:
        min_value = ConstDict(pdict=t_min, scheduler=s)
        max_value = ConstDict(pdict=t_max, scheduler=s)
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
            random = RandomPTable(2, rows=100_000, scheduler=s)
            t_min = PDict({"_1": 0.3})
            t_max = PDict({"_1": 0.8})
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
            random = RandomPTable(2, rows=100_000, scheduler=s)
            t_min = PDict({"_1": 0.0})
            t_max = PDict({"_1": float("nan")})
            range_qry = self._query_min_max_impl(random, t_min, t_max, s)
        aio.run(s.start())
        min_data = cast(PDict, range_qry.output.min.data())
        max_data = range_qry.output.max.data()
        assert random.result is not None
        min_rand = random.result.min()["_1"]
        self.assertAlmostEqual(min_data["_1"], min_rand, delta=0.0001)
        self.assertAlmostEqual(max_data["_1"], 1.0, delta=0.0001)

    def test_range_query_min_max3(self) -> None:
        "Test min and max on RangeQuery output"
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=100_000, scheduler=s)
            t_min = PDict({"_1": 0.3})
            t_max = PDict({"_1": 15000.0})
            range_qry = self._query_min_max_impl(random, t_min, t_max, s)
        aio.run(s.start())
        min_data = cast(PDict, range_qry.output.min.data())
        max_data = cast(PDict, range_qry.output.max.data())
        assert random.result is not None
        max_rand = random.result.max()["_1"]
        self.assertAlmostEqual(min_data["_1"], 0.3)
        self.assertAlmostEqual(max_data["_1"], max_rand)


if __name__ == "__main__":
    ProgressiveTest.main()
