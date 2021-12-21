"Test for Range Query"

from progressivis.table.constant import Constant
from progressivis import Print
from progressivis.stats import RandomTable, Min, Max
from progressivis.core.bitmap import bitmap
from progressivis.table.range_query import RangeQuery
from progressivis.utils.psdict import PsDict
from progressivis.core import aio
from . import ProgressiveTest, main


class TestRangeQuery(ProgressiveTest):
    "Test Suite for RangeQuery Module"

    def tearDown(self):
        TestRangeQuery.cleanup()

    def test_range_query(self):
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
        idx = (
            range_qry.input_module.output["result"]
            .data()
            .eval("(_1>0.3)&(_1<0.8)", result_object="index")
        )
        self.assertEqual(range_qry.result.index, bitmap(idx))

    def test_hist_index_min_max(self):
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
            min_ = Min(name="min_" + str(hash(hist_index)), scheduler=s)
            min_.input[0] = hist_index.output.min_out
            prt2 = Print(proc=self.terse, scheduler=s)
            prt2.input[0] = min_.output.result
            max_ = Max(name="max_" + str(hash(hist_index)), scheduler=s)
            max_.input[0] = hist_index.output.max_out
            pr3 = Print(proc=self.terse, scheduler=s)
            pr3.input[0] = max_.output.result
        aio.run(s.start())
        res1 = random.result.min()["_1"]
        res2 = min_.result["_1"]
        self.assertAlmostEqual(res1, res2)
        res1 = random.result.max()["_1"]
        res2 = max_.result["_1"]
        self.assertAlmostEqual(res1, res2)

    def _query_min_max_impl(self, random, t_min, t_max, s):
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

    def test_range_query_min_max(self):
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

    def test_range_query_min_max2(self):
        "Test min and max on RangeQuery output"
        s = self.scheduler()
        with s:
            random = RandomTable(2, rows=100000, scheduler=s)
            t_min = PsDict({"_1": 0.0})
            t_max = PsDict({"_1": float("nan")})
            range_qry = self._query_min_max_impl(random, t_min, t_max, s)
        aio.run(s.start())
        min_data = range_qry.output.min.data()
        max_data = range_qry.output.max.data()
        min_rand = random.result.min()["_1"]
        self.assertAlmostEqual(min_data["_1"], min_rand, delta=0.0001)
        self.assertAlmostEqual(max_data["_1"], 1.0, delta=0.0001)

    def test_range_query_min_max3(self):
        "Test min and max on RangeQuery output"
        s = self.scheduler()
        with s:
            random = RandomTable(2, rows=100000, scheduler=s)
            t_min = PsDict({"_1": 0.3})
            t_max = PsDict({"_1": 15000.0})
            range_qry = self._query_min_max_impl(random, t_min, t_max, s)
        aio.run(s.start())
        min_data = range_qry.output.min.data()
        max_data = range_qry.output.max.data()
        max_rand = random.result.max()["_1"]
        self.assertAlmostEqual(min_data["_1"], 0.3)
        self.assertAlmostEqual(max_data["_1"], max_rand)


if __name__ == "__main__":
    main()
