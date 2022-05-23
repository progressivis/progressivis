from . import ProgressiveTest, skip
import numpy as np
from progressivis import Print
from progressivis.io import SimpleCSVLoader
from progressivis.stats import Distinct
from progressivis.datasets import get_dataset
from progressivis.core import aio


def print_repr(x):
    print(repr(x))


MIN_INT64 = np.iinfo(np.int64).min


class TestDistinct(ProgressiveTest):
    #    def setUp(self):
    #        log_level(logging.INFO)
    @skip  # TODO fix with a reliable dataset
    def test_distinct_categorical(self):
        s = self.scheduler()
        csv = SimpleCSVLoader(
            get_dataset("nyc_taxis"), index_col=False, nrows=100_000, scheduler=s
        )
        dist = Distinct(scheduler=s)
        dist.input[0] = csv.output.result
        prt = Print(proc=self.terse, scheduler=s)
        prt.input[0] = dist.output.result
        aio.run(csv.scheduler().start())
        res = dist.result
        self.assertEqual(res["passenger_count"], {0, 1, 2, 3, 4, 5, 6, 9})
        self.assertEqual(res["trip_distance"], None)
        self.assertEqual(res["payment_type"], {1, 2, 3, 4})

    def test_distinct_float(self):
        s = self.scheduler()
        csv = SimpleCSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        dist = Distinct(scheduler=s)
        dist.input[0] = csv.output.result
        prt = Print(proc=self.terse, scheduler=s)
        prt.input[0] = dist.output.result
        aio.run(csv.scheduler().start())
        res = [v for v in dist.result.values() if v is not None]
        self.assertEqual(res, [])  # too many values detected in all cols


if __name__ == "__main__":
    ProgressiveTest.main()
