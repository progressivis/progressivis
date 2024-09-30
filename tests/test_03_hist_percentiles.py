"Test for Range Query"
from __future__ import annotations

from progressivis.table.constant import ConstDict
from progressivis import Print
from progressivis.stats import RandomPTable
from progressivis.table.binning_index import BinningIndex
from progressivis.table.percentiles import Percentiles
import numpy as np
from . import ProgressiveTest
from progressivis.table.range_query import RangeQuery
from progressivis.utils.psdict import PDict
from progressivis.table.stirrer import Stirrer
from progressivis.core.api import Sink
from progressivis.core import aio

from typing import Any


class TestPercentiles(ProgressiveTest):

    """Tests for HistIndex based percentiles
    NB: another percentiles module exists in stats directory
    which is based on T-digest
    """

    def tearDown(self) -> None:
        TestPercentiles.cleanup()

    def _impl_tst_percentiles(self, accuracy: float) -> None:
        """ """
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=20_000, scheduler=s)
            hist_index = BinningIndex(scheduler=s)
            hist_index.input[0] = random.output.result["_1",]
            t_percentiles = PDict({"_25": 25.0, "_50": 50.0, "_75": 75.0})
            which_percentiles = ConstDict(pdict=t_percentiles, scheduler=s)
            percentiles = Percentiles(accuracy=accuracy, scheduler=s)
            percentiles.input.percentiles = which_percentiles.output.result
            percentiles.input.index = hist_index.output.result
            sink = Sink(scheduler=s)
            sink.input.inp = percentiles.output.result
        aio.run(s.start())
        assert percentiles.result is not None
        assert random.result is not None
        pdict = percentiles.result
        v = random.result["_1"].values
        p25 = np.percentile(v, 25.0)
        p50 = np.percentile(v, 50.0)
        p75 = np.percentile(v, 75.0)
        print(
            "PTable=> accuracy: ",
            accuracy,
            " 25:",
            p25,
            pdict["_25"],
            " 50:",
            p50,
            pdict["_50"],
            " 75:",
            p75,
            pdict["_75"],
        )
        # from nose.tools import set_trace; set_trace()
        self.assertAlmostEqual(p25, pdict["_25"], delta=0.01)
        self.assertAlmostEqual(p50, pdict["_50"], delta=0.01)
        self.assertAlmostEqual(p75, pdict["_75"], delta=0.01)

    def _impl_stirred_tst_percentiles(self, accuracy: float, **kw: Any) -> None:
        """ """
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=20_000, scheduler=s)
            stirrer = Stirrer(
                update_column="_2", fixed_step_size=1000, scheduler=s, **kw
            )
            stirrer.input[0] = random.output.result
            hist_index = BinningIndex(scheduler=s)
            hist_index.input[0] = stirrer.output.result["_1",]
            t_percentiles = PDict({"_25": 25.0, "_50": 50.0, "_75": 75.0})
            which_percentiles = ConstDict(pdict=t_percentiles, scheduler=s)
            percentiles = Percentiles(accuracy=accuracy, scheduler=s)
            percentiles.input.percentiles = which_percentiles.output.result
            percentiles.input.index = hist_index.output.result
            sink = Sink(scheduler=s)
            sink.input.inp = percentiles.output.result

        aio.run(s.start())
        assert percentiles.result is not None
        assert stirrer.result is not None
        pdict = percentiles.result
        v = stirrer.result.to_array(columns=["_1"]).reshape(-1)
        p25 = np.percentile(v, 25.0)
        p50 = np.percentile(v, 50.0)
        p75 = np.percentile(v, 75.0)
        print(
            "PTable=> accuracy: ",
            accuracy,
            " 25:",
            p25,
            pdict["_25"],
            " 50:",
            p50,
            pdict["_50"],
            " 75:",
            p75,
            pdict["_75"],
        )
        self.assertAlmostEqual(p25, pdict["_25"], delta=0.01)
        self.assertAlmostEqual(p50, pdict["_50"], delta=0.01)
        self.assertAlmostEqual(p75, pdict["_75"], delta=0.01)

    def test_percentiles_fast(self) -> None:
        """test_percentiles: lower  accurracy => faster mode"""
        return self._impl_tst_percentiles(2.0)

    def test_percentiles_fast2(self) -> None:
        """test_percentiles: lower  accurracy => faster mode stirred del"""
        return self._impl_stirred_tst_percentiles(2.0, delete_rows=5)

    def test_percentiles_fast3(self) -> None:
        """test_percentiles: lower  accurracy => faster mode stirred upd"""
        return self._impl_stirred_tst_percentiles(2.0, update_rows=5)

    def test_percentiles_accurate(self) -> None:
        """test_percentiles: higher accurracy => slower mode"""
        return self._impl_tst_percentiles(0.2)

    def test_percentiles_accurate2(self) -> None:
        """test_percentiles: higher accurracy => slower mode stirred del"""
        return self._impl_stirred_tst_percentiles(0.2, delete_rows=5)

    def test_percentiles_accurate3(self) -> None:
        """test_percentiles: higher accurracy => slower mode stirred upd"""
        return self._impl_stirred_tst_percentiles(0.2, update_rows=5)

    def _impl_tst_percentiles_rq(self, accuracy: float) -> None:
        """ """
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=50_000, scheduler=s)
            t_min = PDict({"_1": 0.3})
            min_value = ConstDict(pdict=t_min, scheduler=s)
            t_max = PDict({"_1": 0.8})
            max_value = ConstDict(pdict=t_max, scheduler=s)
            range_qry = RangeQuery(column="_1", scheduler=s)
            range_qry.create_dependent_modules(
                random, "result", min_value=min_value, max_value=max_value
            )

            hist_index = BinningIndex(scheduler=s)
            hist_index.input.table = range_qry.output.result["_1",]
            t_percentiles = PDict({"_25": 25.0, "_50": 50.0, "_75": 75.0})
            which_percentiles = ConstDict(pdict=t_percentiles, scheduler=s)
            percentiles = Percentiles(accuracy=accuracy, scheduler=s)
            percentiles.input.percentiles = which_percentiles.output.result
            percentiles.input.index = hist_index.output.result
            sink = Sink(scheduler=s)
            sink.input.inp = percentiles.output.result
        aio.run(s.start())
        assert percentiles.result is not None
        assert range_qry.result is not None
        pdict = percentiles.result
        v = range_qry.result["_1"].values
        p25 = np.percentile(v, 25.0)
        p50 = np.percentile(v, 50.0)
        p75 = np.percentile(v, 75.0)
        print(
            "TSV=> accuracy: ",
            accuracy,
            " 25:",
            p25,
            pdict["_25"],
            " 50:",
            p50,
            pdict["_50"],
            " 75:",
            p75,
            pdict["_75"],
        )
        # from nose.tools import set_trace; set_trace()
        self.assertAlmostEqual(p25, pdict["_25"], delta=0.01)
        self.assertAlmostEqual(p50, pdict["_50"], delta=0.01)
        self.assertAlmostEqual(p75, pdict["_75"], delta=0.01)

    def _impl_stirred_tst_percentiles_rq(self, accuracy: float, **kw: Any) -> None:
        """ """
        s = self.scheduler()
        with s:
            random = RandomPTable(2, rows=20_000, scheduler=s)
            stirrer = Stirrer(
                update_column="_2", fixed_step_size=1000, scheduler=s, **kw
            )
            stirrer.input[0] = random.output.result
            t_min = PDict({"_1": 0.3})
            min_value = ConstDict(pdict=t_min, scheduler=s)
            t_max = PDict({"_1": 0.8})
            max_value = ConstDict(pdict=t_max, scheduler=s)
            range_qry = RangeQuery(column="_1", scheduler=s)
            range_qry.create_dependent_modules(
                stirrer, "result", min_value=min_value, max_value=max_value
            )

            hist_index = BinningIndex(scheduler=s)
            hist_index.input[0] = range_qry.output.result["_1",]
            t_percentiles = PDict({"_25": 25.0, "_50": 50.0, "_75": 75.0})
            which_percentiles = ConstDict(pdict=t_percentiles, scheduler=s)
            percentiles = Percentiles(accuracy=accuracy, scheduler=s)
            percentiles.input.percentiles = which_percentiles.output.result
            percentiles.input.index = hist_index.output.result
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = percentiles.output.result
        aio.run(s.start())
        assert percentiles.result is not None
        assert range_qry.result is not None
        pdict = percentiles.result
        v = range_qry.result["_1"].values
        p25 = np.percentile(v, 25.0)
        p50 = np.percentile(v, 50.0)
        p75 = np.percentile(v, 75.0)
        print(
            "TSV=> accuracy: ",
            accuracy,
            " 25:",
            p25,
            pdict["_25"],
            " 50:",
            p50,
            pdict["_50"],
            " 75:",
            p75,
            pdict["_75"],
        )
        self.assertAlmostEqual(p25, pdict["_25"], delta=0.01)
        self.assertAlmostEqual(p50, pdict["_50"], delta=0.01)
        self.assertAlmostEqual(p75, pdict["_75"], delta=0.01)

    def test_percentiles_fast_rq(self) -> None:
        """test_percentiles: lower  accurracy => faster mode rq"""
        return self._impl_tst_percentiles_rq(2.0)

    def test_percentiles_fast_rq2(self) -> None:
        """test_percentiles: lower  accurracy => faster mode rq stirred del"""
        return self._impl_stirred_tst_percentiles_rq(2.0, delete_rows=5)

    def test_percentiles_fast_rq3(self) -> None:
        """test_percentiles: lower  accurracy => faster mode rq stirred upd"""
        return self._impl_stirred_tst_percentiles_rq(2.0, update_rows=5)

    def test_percentiles_accurate_rq(self) -> None:
        """test_percentiles: higher accurracy => slower mode rq"""
        return self._impl_tst_percentiles_rq(0.2)

    def test_percentiles_accurate_rq2(self) -> None:
        """test_percentiles: higher accurracy => slower mode rq stirred del"""
        return self._impl_stirred_tst_percentiles_rq(0.2, delete_rows=5)

    def test_percentiles_accurate_rq3(self) -> None:
        """test_percentiles: higher accurracy => slower mode rq stirred upd"""
        return self._impl_stirred_tst_percentiles_rq(0.2, update_rows=5)


if __name__ == "__main__":
    ProgressiveTest.main()
