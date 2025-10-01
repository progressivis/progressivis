from __future__ import annotations

import os
from . import ProgressiveTest, skipIf
from progressivis import Sink, ParquetLoader, PTable, get_dataset
from progressivis.core import aio
from progressivis.table.group_by import GroupBy, ByType, SubPColumn as SC
from progressivis.table.aggregate import Aggregate
from progressivis.table.stirrer import Stirrer

import pyarrow.parquet as pq
import numpy as np
from typing import Any, Tuple

# PARQUET_FILE = "nyc-taxi/short_500k_yellow_tripdata_2015-01.parquet"
# PARQUET_FILE = "../nyc-taxi/short_500k_yellow_tripdata_2015-01.parquet"
PARQUET_FILE = get_dataset("short-taxis2015-01_parquet")

# NB: if PARQUET_FILE does not exist yet, consider running:
# python scripts/create_nyc_parquet.py -p short -t yellow -f -m1 -n 300000
if not os.getenv("CI"):
    TABLE = pq.read_table(PARQUET_FILE)
    TABLE_AGGR = TABLE.group_by("passenger_count").aggregate(  # type: ignore
        [("trip_distance", "mean"),
         ("trip_distance", "sum"),
         ("RateCodeID", "count_distinct"),]
    )
    TABLE_AGGR_2 = TABLE.group_by("passenger_count").aggregate(  # type: ignore
        [("trip_distance", "mean"),
         ("trip_distance", "sum")]
    )
    TABLE_AGGR_3 = TABLE.group_by("passenger_count").aggregate(  # type: ignore
        [("fare_amount", "mean"),
         ("trip_distance", "mean"),
         ("trip_distance", "sum")]
    )
    TABLE_AGGR_4 = TABLE.group_by(["passenger_count", "VendorID"]).aggregate(  # type: ignore
        [("trip_distance", "mean")]
    )

    DF = TABLE.to_pandas()  # type: ignore
    DF_AGGR = DF.groupby(DF.tpep_pickup_datetime.dt.day).trip_distance.mean()


@skipIf(os.getenv("CI"), "skipped because local nyc taxi files are required")
class TestProgressiveAggregate(ProgressiveTest):
    def test_aggregate_1_col(self) -> None:
        s = self.scheduler
        parquet = ParquetLoader(
            PARQUET_FILE, columns=["passenger_count", "RateCodeID"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None, "parquet.result is not None")
        grby = GroupBy(by="passenger_count", scheduler=s)
        grby.input.table = parquet.output.result
        # aggr = Aggregate(compute=[("trip_distance", "mean")], scheduler=s)
        aggr = Aggregate(compute=[("RateCodeID", "uniq")], scheduler=s)
        aggr.input.table = grby.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = aggr.output.result
        aio.run(s.start())
        assert aggr.result is not None
        self.assertTrue(
            np.array_equal(
                aggr.result["passenger_count"].value,
                TABLE_AGGR["passenger_count"].to_numpy(),
            ),
            "passenger_count is not equal"
        )
        self.assertTrue(
            np.array_equal(
                aggr.result["RateCodeID_uniq"].value,
                TABLE_AGGR["RateCodeID_count_distinct"].to_numpy()
            ),
            "RateCodeID_count_distinct is not equal"
        )

    def test_aggregate_1_col_delete(self) -> None:
        s = self.scheduler
        removed = 11142
        parquet = ParquetLoader(
            PARQUET_FILE, columns=["passenger_count", "trip_distance"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None)
        stirrer = Stirrer(
            update_column="trip_distance",
            delete_rows=[removed],
            # update_rows=5,
            fixed_step_size=1000,
            scheduler=s,
        )
        stirrer.input[0] = parquet.output.result
        grby = GroupBy(by="passenger_count", scheduler=s)
        grby.input.table = stirrer.output.result
        aggr = Aggregate(compute=[("trip_distance", "sum")], scheduler=s)
        aggr.input.table = grby.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = aggr.output.result
        aio.run(s.start())
        assert aggr.result is not None
        self.assertTrue(
            np.array_equal(
                aggr.result["passenger_count"].value,
                TABLE_AGGR["passenger_count"].to_numpy(),
            ),
            "passenger_count is not equal"
        )
        self.assertTrue(
            np.allclose(
                sum(TABLE_AGGR["trip_distance_sum"].to_numpy())
                - sum(aggr.result.loc[:, "trip_distance_sum"].to_array()),
                TABLE["trip_distance"][removed].as_py(),  # type: ignore
            ),
            "trip_distance_sum is not equal"
        )

    def test_aggregate_1_col_update(self) -> None:
        s = self.scheduler
        upd_id = 1142
        new_val = 10.0
        parquet = ParquetLoader(
            PARQUET_FILE, columns=["passenger_count", "trip_distance"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None)
        stirrer = Stirrer(
            update_column="trip_distance",
            update_rows=([upd_id], [new_val]),
            fixed_step_size=1000,
            scheduler=s,
        )
        stirrer.input[0] = parquet.output.result
        grby = GroupBy(by="passenger_count", scheduler=s)
        grby.input.table = stirrer.output.result
        aggr = Aggregate(compute=[("trip_distance", "sum")], scheduler=s)
        aggr.input.table = grby.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = aggr.output.result
        aio.run(s.start())
        assert aggr.result is not None
        self.assertTrue(
            np.array_equal(
                aggr.result["passenger_count"].value,
                TABLE_AGGR["passenger_count"].to_numpy(),
            ),
            "passenger_count is not equal"
        )
        self.assertTrue(
            np.allclose(
                abs(
                    sum(TABLE_AGGR["trip_distance_sum"].to_numpy())
                    - sum(aggr.result.loc[:, "trip_distance_sum"].to_array())
                ),
                abs(new_val - TABLE["trip_distance"][upd_id].as_py()),  # type: ignore
            ),
            "trip_distance_sum is not close"
        )

    def test_aggregate_2_col(self) -> None:
        s = self.scheduler
        parquet = ParquetLoader(
            PARQUET_FILE, columns=["passenger_count", "trip_distance"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None)
        grby = GroupBy(by="passenger_count", scheduler=s)
        grby.input.table = parquet.output.result
        aggr = Aggregate(compute=[("trip_distance", "mean")], scheduler=s)
        aggr.input.table = grby.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = aggr.output.result
        aio.run(s.start())
        assert aggr.result is not None
        self.assertTrue(
            np.array_equal(
                aggr.result["passenger_count"].value,
                TABLE_AGGR["passenger_count"].to_numpy(),
            )
        )
        self.assertTrue(
            np.allclose(
                aggr.result["trip_distance_mean"].value,
                TABLE_AGGR["trip_distance_mean"].to_numpy(),
            )
        )

    def test_aggregate_2_fnc(self) -> None:
        s = self.scheduler
        parquet = ParquetLoader(
            PARQUET_FILE, columns=["passenger_count", "trip_distance"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None)
        grby = GroupBy(by="passenger_count", scheduler=s)
        grby.input.table = parquet.output.result
        aggr = Aggregate(
            compute=[("trip_distance", "mean"), ("trip_distance", "sum")],
            scheduler=s
        )
        aggr.input.table = grby.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = aggr.output.result
        aio.run(s.start())
        assert aggr.result is not None
        self.assertTrue(
            np.array_equal(
                aggr.result["passenger_count"].value,
                TABLE_AGGR_2["passenger_count"].to_numpy(),
            )
        )
        self.assertTrue(
            np.allclose(
                aggr.result["trip_distance_mean"].value,
                TABLE_AGGR_2["trip_distance_mean"].to_numpy(),
            )
        )
        self.assertTrue(
            np.allclose(
                aggr.result["trip_distance_sum"].value,
                TABLE_AGGR_2["trip_distance_sum"].to_numpy(),
            )
        )

    def test_aggregate_3_fnc(self) -> None:
        s = self.scheduler
        parquet = ParquetLoader(
            PARQUET_FILE,
            columns=["fare_amount", "passenger_count", "trip_distance"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None)
        grby = GroupBy(by="passenger_count", scheduler=s)
        grby.input.table = parquet.output.result
        aggr = Aggregate(
            compute=[
                ("fare_amount", "mean"),
                ("trip_distance", "mean"),
                ("trip_distance", "sum"),
            ],
            scheduler=s,
        )
        aggr.input.table = grby.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = aggr.output.result
        aio.run(s.start())
        assert aggr.result is not None
        self.assertTrue(
            np.array_equal(
                aggr.result["passenger_count"].value,
                TABLE_AGGR_3["passenger_count"].to_numpy(),
            ),
            "passenger_count is different"
        )
        self.assertTrue(
            np.allclose(
                aggr.result["fare_amount_mean"].value,
                TABLE_AGGR_3["fare_amount_mean"].to_numpy(),
            ),
            "fare_amount_mean is different"
        )
        self.assertTrue(
            np.allclose(
                aggr.result["trip_distance_mean"].value,
                TABLE_AGGR_3["trip_distance_mean"].to_numpy(),
            ),
            "trip_distance_mean is different"
        )
        self.assertTrue(
            np.allclose(
                aggr.result["trip_distance_sum"].value,
                TABLE_AGGR_3["trip_distance_sum"].to_numpy(),
            ),
            "trip_distance_sum is different"
        )

    def test_aggregate_2_groups(self) -> None:
        s = self.scheduler
        parquet = ParquetLoader(
            PARQUET_FILE,
            columns=["passenger_count", "VendorID", "trip_distance"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None)
        grby = GroupBy(by=["passenger_count", "VendorID"], scheduler=s)
        grby.input.table = parquet.output.result
        aggr = Aggregate(compute=[("trip_distance", "mean")], scheduler=s)
        aggr.input.table = grby.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = aggr.output.result
        aio.run(s.start())
        assert aggr.result is not None
        self.assertTrue(
            np.array_equal(
                aggr.result["passenger_count"].value,
                TABLE_AGGR_4["passenger_count"].to_numpy(),
            )
        )
        self.assertTrue(
            np.array_equal(
                aggr.result["VendorID"].value,
                TABLE_AGGR_4["VendorID"].to_numpy()
            )
        )
        self.assertTrue(
            np.allclose(
                aggr.result["trip_distance_mean"].value,
                TABLE_AGGR_4["trip_distance_mean"].to_numpy(),
            )
        )

    def test_aggregate_by_func(self) -> None:
        s = self.scheduler
        parquet = ParquetLoader(
            PARQUET_FILE,
            columns=["tpep_pickup_datetime", "trip_distance"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None)

        def _day_func(tbl: PTable, i: int) -> Tuple[Any, ...]:
            dt = tbl.loc[i, "tpep_pickup_datetime"]
            assert dt is not None
            return tuple(dt[:3])

        grby = GroupBy(by=_day_func, scheduler=s)
        grby.input.table = parquet.output.result
        aggr = Aggregate(compute=[("trip_distance", "mean")], scheduler=s)
        aggr.input.table = grby.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = aggr.output.result
        aio.run(s.start())
        assert aggr.result is not None
        self.assertTrue(
            np.allclose(
                sorted(aggr.result["trip_distance_mean"].value),
                sorted(DF_AGGR.values)
            )
        )

    def _aggregate_by_subcolumn(self, subcol: ByType) -> None:
        s = self.scheduler
        parquet = ParquetLoader(
            PARQUET_FILE,
            columns=["tpep_pickup_datetime", "trip_distance"],
            scheduler=s,
        )
        self.assertTrue(parquet.result is None)
        grby = GroupBy(by=subcol, scheduler=s)
        grby.input.table = parquet.output.result
        aggr = Aggregate(compute=[("trip_distance", "mean")], scheduler=s)
        aggr.input.table = grby.output.result
        sink = Sink(scheduler=s)
        sink.input.inp = aggr.output.result
        aio.run(s.start())
        assert aggr.result is not None
        self.assertTrue(
            np.allclose(
                sorted(aggr.result["trip_distance_mean"].value),
                sorted(DF_AGGR.values)
            )
        )

    def test_aggregate_by_subcolumn_fancy(self) -> None:
        return self._aggregate_by_subcolumn(SC("tpep_pickup_datetime").ix[[0, 1, 2]])

    def test_aggregate_by_subcolumn_slice(self) -> None:
        return self._aggregate_by_subcolumn(SC("tpep_pickup_datetime").ix[:3])

    def test_aggregate_by_subcolumn_datetime_ix(self) -> None:
        return self._aggregate_by_subcolumn(SC("tpep_pickup_datetime").dt["YMD"])  # YMDhms

    def test_aggregate_by_subcolumn_datetime_dt(self) -> None:
        return self._aggregate_by_subcolumn(SC("tpep_pickup_datetime").dt.year.month.day)


if __name__ == "__main__":
    ProgressiveTest.main()
