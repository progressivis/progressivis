from __future__ import annotations

import os
from . import ProgressiveTest, skipIf
from progressivis.core import aio, Sink
from progressivis.io import ParquetLoader
from progressivis.table.group_by import GroupBy

PASSENGERS = {0, 1, 2, 3, 4, 5, 6, 9}

# NB: if the file does not exist, consider running: python scripts/create_nyc_parquet.py -p short -t yellow -f -m1 -n 300000


@skipIf(os.getenv("CI"), "skipped because local nyc taxi files are required")
class TestProgressiveGroupBy(ProgressiveTest):
    def test_group_by_1_col(self) -> None:
        s = self.scheduler()
        parquet = ParquetLoader(
            "nyc-taxi/short_500k_yellow_tripdata_2015-01.parquet",
            columns=["passenger_count", "extra", "trip_distance"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None)
        grby = GroupBy(by="passenger_count", scheduler=s)
        grby.input.table = parquet.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = grby.output.result
        aio.run(s.start())
        self.assertEqual(len(parquet.table), 300_000)
        self.assertEqual(set(grby._index.keys()), PASSENGERS)

    def test_group_by_2_cols(self) -> None:
        s = self.scheduler()
        parquet = ParquetLoader(
            "nyc-taxi/short_500k_yellow_tripdata_2015-01.parquet",
            columns=["passenger_count", "extra", "trip_distance"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None)
        grby = GroupBy(by=["passenger_count", "extra"], scheduler=s)
        grby.input.table = parquet.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = grby.output.result
        aio.run(s.start())
        self.assertEqual(len(parquet.table), 300_000)
        self.assertEqual(len(grby._index.keys()), 36)

    def test_group_by_function(self) -> None:
        s = self.scheduler()
        parquet = ParquetLoader(
            "nyc-taxi/short_500k_yellow_tripdata_2015-01.parquet",
            columns=["passenger_count", "extra", "trip_distance"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None)
        grby = GroupBy(by=lambda t, i: i % 10, scheduler=s)
        grby.input.table = parquet.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = grby.output.result
        aio.run(s.start())
        self.assertEqual(len(parquet.table), 300_000)
        self.assertEqual(len(grby._index.keys()), 10)

    def test_group_by_days(self) -> None:
        s = self.scheduler()
        parquet = ParquetLoader(
            "nyc-taxi/short_500k_yellow_tripdata_2015-01.parquet",
            columns=[
                "tpep_dropoff_datetime",
                "passenger_count",
                "extra",
                "trip_distance",
            ],
            scheduler=s,
        )

        def _day_func(tbl, i):
            dt = tbl.loc[i, "tpep_dropoff_datetime"]
            return tuple(dt[:3])

        self.assertTrue(parquet.result is None)
        grby = GroupBy(by=_day_func, scheduler=s)
        grby.input.table = parquet.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = grby.output.result
        aio.run(s.start())
        self.assertEqual(len(parquet.table), 300_000)
        self.assertEqual(
            len(grby._index.keys()), 32
        )  # the file contains few february records ...


if __name__ == "__main__":
    ProgressiveTest.main()
