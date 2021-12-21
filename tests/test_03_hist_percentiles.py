"Test for Range Query"
from progressivis.core import aio
from progressivis.table.constant import Constant
from progressivis import Print
from progressivis.stats import RandomTable
from progressivis.table.hist_index import HistogramIndex
from progressivis.table.percentiles import Percentiles
import numpy as np
from . import ProgressiveTest, main
from progressivis.table.range_query import RangeQuery
from progressivis.utils.psdict import PsDict
from progressivis.table.stirrer import Stirrer


class TestPercentiles(ProgressiveTest):

    """Tests for HistIndex based percentiles
    NB: another percentiles module exists in stats directory
    which is based on T-digest
    """

    def tearDown(self):
        TestPercentiles.cleanup()

    def _impl_tst_percentiles(self, accuracy):
        """
        """
        s = self.scheduler()
        with s:
            random = RandomTable(2, rows=10000, scheduler=s)
            hist_index = HistogramIndex(column="_1", scheduler=s)
            hist_index.input[0] = random.output.result
            t_percentiles = PsDict({"_25": 25.0, "_50": 50.0, "_75": 75.0})
            which_percentiles = Constant(table=t_percentiles, scheduler=s)
            percentiles = Percentiles(hist_index, accuracy=accuracy, scheduler=s)
            percentiles.input[0] = random.output.result
            percentiles.input.percentiles = which_percentiles.output.result
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = percentiles.output.result
        aio.run(s.start())
        pdict = percentiles.result.last().to_dict()
        v = random.result["_1"].values
        p25 = np.percentile(v, 25.0)
        p50 = np.percentile(v, 50.0)
        p75 = np.percentile(v, 75.0)
        print(
            "Table=> accuracy: ",
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

    def _impl_stirred_tst_percentiles(self, accuracy, **kw):
        """
        """
        s = self.scheduler()
        with s:
            random = RandomTable(2, rows=10000, scheduler=s)
            stirrer = Stirrer(
                update_column="_2", fixed_step_size=1000, scheduler=s, **kw
            )
            stirrer.input[0] = random.output.result
            hist_index = HistogramIndex(column="_1", scheduler=s)
            hist_index.input[0] = stirrer.output.result
            t_percentiles = PsDict({"_25": 25.0, "_50": 50.0, "_75": 75.0})
            which_percentiles = Constant(table=t_percentiles, scheduler=s)
            percentiles = Percentiles(hist_index, accuracy=accuracy, scheduler=s)
            percentiles.input[0] = stirrer.output.result
            percentiles.input.percentiles = which_percentiles.output.result
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = percentiles.output.result
        aio.run(s.start())
        pdict = percentiles.result.last().to_dict()
        # v = random.table()['_1'].values
        # import pdb;pdb.set_trace()
        v = stirrer.result.to_array(columns=["_1"]).reshape(-1)
        p25 = np.percentile(v, 25.0)
        p50 = np.percentile(v, 50.0)
        p75 = np.percentile(v, 75.0)
        print(
            "Table=> accuracy: ",
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

    def test_percentiles_fast(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        low accurracy => faster mode
        """
        return self._impl_tst_percentiles(2.0)

    def test_percentiles_fast2(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        low accurracy => faster mode
        """
        return self._impl_stirred_tst_percentiles(2.0, delete_rows=5)

    def test_percentiles_fast3(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        low accurracy => faster mode
        """
        return self._impl_stirred_tst_percentiles(2.0, update_rows=5)

    def test_percentiles_accurate(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        higher accurracy => slower mode
        """
        return self._impl_tst_percentiles(0.2)

    def test_percentiles_accurate2(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        higher accurracy => slower mode
        """
        return self._impl_stirred_tst_percentiles(0.2, delete_rows=5)

    def test_percentiles_accurate3(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        higher accurracy => slower mode
        """
        return self._impl_stirred_tst_percentiles(0.2, update_rows=5)

    def _impl_tst_percentiles_rq(self, accuracy):
        """
        """
        s = self.scheduler()
        with s:
            random = RandomTable(2, rows=10000, scheduler=s)
            t_min = PsDict({"_1": 0.3})
            min_value = Constant(table=t_min, scheduler=s)
            t_max = PsDict({"_1": 0.8})
            max_value = Constant(table=t_max, scheduler=s)
            range_qry = RangeQuery(column="_1", scheduler=s)
            range_qry.create_dependent_modules(
                random, "result", min_value=min_value, max_value=max_value
            )

            hist_index = range_qry.hist_index
            t_percentiles = PsDict({"_25": 25.0, "_50": 50.0, "_75": 75.0})
            which_percentiles = Constant(table=t_percentiles, scheduler=s)
            percentiles = Percentiles(hist_index, accuracy=accuracy, scheduler=s)
            percentiles.input[0] = range_qry.output.result
            percentiles.input.percentiles = which_percentiles.output.result
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = percentiles.output.result
        aio.run(s.start())
        pdict = percentiles.result.last().to_dict()
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

    def _impl_stirred_tst_percentiles_rq(self, accuracy, **kw):
        """
        """
        s = self.scheduler()
        with s:
            random = RandomTable(2, rows=10000, scheduler=s)
            stirrer = Stirrer(
                update_column="_2", fixed_step_size=1000, scheduler=s, **kw
            )
            stirrer.input[0] = random.output.result
            t_min = PsDict({"_1": 0.3})
            min_value = Constant(table=t_min, scheduler=s)
            t_max = PsDict({"_1": 0.8})
            max_value = Constant(table=t_max, scheduler=s)
            range_qry = RangeQuery(column="_1", scheduler=s)
            range_qry.create_dependent_modules(
                stirrer, "result", min_value=min_value, max_value=max_value
            )

            hist_index = range_qry.hist_index
            t_percentiles = PsDict({"_25": 25.0, "_50": 50.0, "_75": 75.0})
            which_percentiles = Constant(table=t_percentiles, scheduler=s)
            percentiles = Percentiles(hist_index, accuracy=accuracy, scheduler=s)
            percentiles.input[0] = range_qry.output.result
            percentiles.input.percentiles = which_percentiles.output.result
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = percentiles.output.result
        aio.run(s.start())
        pdict = percentiles.result.last().to_dict()
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

    def test_percentiles_fast_rq(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        low accurracy => faster mode
        """
        return self._impl_tst_percentiles_rq(2.0)

    def test_percentiles_fast_rq2(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        low accurracy => faster mode
        """
        return self._impl_stirred_tst_percentiles_rq(2.0, delete_rows=5)

    def test_percentiles_fast_rq3(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        low accurracy => faster mode
        """
        return self._impl_stirred_tst_percentiles_rq(2.0, update_rows=5)

    def test_percentiles_accurate_rq(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        higher accurracy => slower mode
        """
        return self._impl_tst_percentiles_rq(0.2)

    def test_percentiles_accurate_rq2(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        higher accurracy => slower mode
        """
        return self._impl_stirred_tst_percentiles_rq(0.2, delete_rows=5)

    def test_percentiles_accurate_rq3(self):
        """test_percentiles: Simple test for HistIndex based percentiles
        higher accurracy => slower mode
        """
        return self._impl_stirred_tst_percentiles_rq(0.2, update_rows=5)


if __name__ == "__main__":
    main()
