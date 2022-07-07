from __future__ import annotations

import os
from . import ProgressiveTest, skipIf
from progressivis.core import aio, Sink
from progressivis.io import ParquetLoader, SimpleCSVLoader
from progressivis.table.join import Join
import pandas as pd
import numpy as np

PARQUET_FILE = "nyc-taxi/newstyle_500k_yellow_tripdata_2015-01.parquet"
# CSV_URL = "https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv"
CSV_URL = "nyc-taxi/taxi+_zone_lookup.csv"
# NB: if PARQUET_FILE does not exist yet, consider running:
# python scripts/create_nyc_parquet.py -p newstyle -t yellow -m1 -n 300000
if not os.getenv("CI"):
    TAXI_COLS = [
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "store_and_fwd_flag",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "fare_amount",
    ]
    TAXI_COLS_LESS = [
        "VendorID",
        "passenger_count",
        "trip_distance",
        "RatecodeID",
        "store_and_fwd_flag",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "fare_amount",
    ]
    TAXIS = pd.read_parquet(PARQUET_FILE, columns=TAXI_COLS)  # type: ignore
    TAXIS["control_id"] = range(len(TAXIS))
    LOOKUP_SKIP_ROWS = [3, 4, 263, 264, 265]
    LOOKUP = pd.read_csv(CSV_URL, skiprows=LOOKUP_SKIP_ROWS,)
    LOOKUP["lookup_id"] = range(len(LOOKUP))
    INNER = TAXIS.join(
        LOOKUP.set_index("LocationID"), on="DOLocationID", how="inner"
    ).sort_values(["control_id"])
    INNER_PU = INNER.join(
        LOOKUP.set_index("LocationID"), on="PULocationID", how="inner", rsuffix="_pu"
    ).sort_values(["control_id"])
    LEFT_OUTER = (
        TAXIS.join(LOOKUP.set_index("LocationID"), on="DOLocationID", how="left")
        .sort_values(["control_id"])
        .fillna(0)
    )
    LEFT_OUTER_PU = (
        LEFT_OUTER.join(
            LOOKUP.set_index("LocationID"), on="PULocationID", how="left", rsuffix="_pu"
        )
        .sort_values(["control_id"])
        .fillna(0)
    )
    RIGHT_OUTER = (
        TAXIS.join(LOOKUP.set_index("LocationID"), on="DOLocationID", how="right")
        .sort_values(["control_id"])
        .fillna(0)
    )
    OUTER_NA = TAXIS.join(
        LOOKUP.set_index("LocationID"), on="DOLocationID", how="outer"
    ).sort_values("DOLocationID")
    OUTER = OUTER_NA.fillna(0)
    OUTER_PU = OUTER_NA.join(
        LOOKUP.set_index("LocationID"), on="PULocationID", how="outer", rsuffix="_pu"
    ).sort_values("PULocationID")
    OUTER_PU2 = OUTER.join(
        LOOKUP.set_index("LocationID"), on="PULocationID", how="outer", rsuffix="_pu"
    ).sort_values("PULocationID")


@skipIf(os.getenv("CI"), "skipped because local nyc taxi files are required")
class TestProgressiveJoin(ProgressiveTest):
    def test_inner(self) -> None:
        s = self.scheduler()
        parquet = ParquetLoader(PARQUET_FILE, columns=TAXI_COLS, scheduler=s,)
        csv = SimpleCSVLoader(CSV_URL, skiprows=LOOKUP_SKIP_ROWS, scheduler=s,)
        self.assertTrue(parquet.result is None)
        join = Join(how="inner", scheduler=s)
        join.create_dependent_modules(
            related_module=parquet,
            primary_module=csv,
            related_on="DOLocationID",
            primary_on="LocationID",
            related_cols=TAXI_COLS,
        )
        sink = Sink(scheduler=s)
        sink.input.inp = join.output.result
        aio.run(s.start())
        df = join.table.to_df(
            to_datetime=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        )
        self.assertEqual(len(df), len(INNER))
        self.assertEqual(
            set(df.columns) | {"control_id", "lookup_id"}, set(INNER.columns)
        )
        for col in df.columns:
            self.assertTrue(np.array_equal(df[col].values, INNER[col].values))

    def test_outer(self) -> None:
        s = self.scheduler()
        parquet = ParquetLoader(PARQUET_FILE, columns=TAXI_COLS, scheduler=s,)
        csv = SimpleCSVLoader(CSV_URL, skiprows=LOOKUP_SKIP_ROWS, scheduler=s,)
        self.assertTrue(parquet.result is None)
        join = Join(how="outer", fillna=0, scheduler=s)
        join.create_dependent_modules(
            related_module=parquet,
            primary_module=csv,
            related_on="DOLocationID",
            primary_on="LocationID",
            related_cols=TAXI_COLS,
        )
        sink = Sink(scheduler=s)
        sink.input.inp = join.output.result
        sink2 = Sink(scheduler=s)
        sink2.input.inp = join.output.primary_outer
        aio.run(s.start())
        df = join.table.to_df(
            to_datetime=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        )
        self.assertEqual(len(df), len(LEFT_OUTER))
        self.assertEqual(
            set(df.columns) | {"control_id", "lookup_id"}, set(LEFT_OUTER.columns)
        )
        for col in df.columns:
            self.assertTrue(np.array_equal(df[col].values, LEFT_OUTER[col].values))
        df2 = join._primary_outer.to_df().rename(columns={"LocationID": "DOLocationID"})
        df_concat = (
            pd.concat([df[df2.columns], df2])
            .set_index("DOLocationID")
            .sort_values("DOLocationID")
        )
        for col in df_concat.columns:
            self.assertTrue(np.array_equal(df_concat[col].values, OUTER[col].values))

    def test_inner_pu(self) -> None:
        s = self.scheduler()
        parquet = ParquetLoader(PARQUET_FILE, columns=TAXI_COLS, scheduler=s,)
        csv = SimpleCSVLoader(CSV_URL, skiprows=LOOKUP_SKIP_ROWS, scheduler=s,)
        self.assertTrue(parquet.result is None)
        join = Join(how="inner", scheduler=s)
        join.create_dependent_modules(
            related_module=parquet,
            primary_module=csv,
            related_on="DOLocationID",
            primary_on="LocationID",
            related_cols=TAXI_COLS,
        )
        join_pu = Join(how="inner", scheduler=s)
        join_pu.create_dependent_modules(
            related_module=join,
            primary_module=csv,
            related_on="PULocationID",
            primary_on="LocationID",
            related_cols=TAXI_COLS + list(LOOKUP.columns),
            suffix="_pu",
        )
        sink = Sink(scheduler=s)
        sink.input.inp = join_pu.output.result
        aio.run(s.start())
        df = join_pu.table.to_df(
            to_datetime=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        )
        self.assertEqual(len(df), len(INNER_PU))
        self.assertEqual(
            set(df.columns) | {"control_id", "lookup_id", "lookup_id_pu"},
            set(INNER_PU.columns),
        )
        for col in df.columns:
            self.assertTrue(np.array_equal(df[col].values, INNER_PU[col].values))

    def test_outer_pu(self) -> None:
        s = self.scheduler()
        parquet = ParquetLoader(PARQUET_FILE, columns=TAXI_COLS, scheduler=s,)
        csv = SimpleCSVLoader(CSV_URL, skiprows=LOOKUP_SKIP_ROWS, scheduler=s,)
        self.assertTrue(parquet.result is None)
        join = Join(how="outer", fillna=0, scheduler=s)
        join.create_dependent_modules(
            related_module=parquet,
            primary_module=csv,
            related_on="DOLocationID",
            primary_on="LocationID",
            related_cols=TAXI_COLS,
        )
        join_pu = Join(how="outer", fillna=0, scheduler=s)
        join_pu.create_dependent_modules(
            related_module=join,
            primary_module=csv,
            related_on="PULocationID",
            primary_on="LocationID",
            related_cols=TAXI_COLS + list(LOOKUP.columns),
            suffix="_pu",
        )
        sink = Sink(scheduler=s)
        sink.input.inp = join_pu.output.result
        sink2 = Sink(scheduler=s)
        sink2.input.inp = join.output.primary_outer
        sink3 = Sink(scheduler=s)
        sink3.input.inp = join_pu.output.primary_outer
        aio.run(s.start())
        df = join_pu.table.to_df(
            to_datetime=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        )
        self.assertEqual(len(df), len(LEFT_OUTER_PU))
        self.assertEqual(
            set(df.columns) | {"control_id", "lookup_id", "lookup_id_pu"},
            set(LEFT_OUTER_PU.columns),
        )
        for col in df.columns:
            self.assertTrue(np.array_equal(df[col].values, LEFT_OUTER_PU[col].values))
        df2 = join._primary_outer.to_df().rename(columns={"LocationID": "PULocationID"})
        df3 = join_pu._primary_outer.to_df().rename(
            columns={"LocationID": "PULocationID"}
        )
        view = list(df2.columns)[1:]  # i.e. 'Borough', 'Zone', 'service_zone'
        inner_ = OUTER_PU.query("VendorID==VendorID")
        outer_pu_ = OUTER_PU.query("VendorID!=VendorID and Zone!=Zone")
        outer_1_ = OUTER_PU.query("VendorID!=VendorID and Zone_pu!=Zone_pu")
        self.assertEqual(len(inner_), len(df))
        self.assertEqual(len(outer_1_), len(df2))
        self.assertEqual(len(outer_pu_), len(df3))
        ord = ["control_id", "lookup_id", "lookup_id_pu"]
        self.assertTrue(inner_.sort_values(ord)[view].fillna(0).equals(df[view]))
        self.assertTrue(
            outer_1_.sort_values(ord)[view]
            .fillna(0)
            .set_index(df2.index)
            .equals(df2[view])
        )
        view_pu = [f"{c}_pu" for c in view]
        outer_pu__ = outer_pu_.sort_values(ord)[view_pu].fillna(0).set_index(df3.index)
        outer_pu__.columns = view
        self.assertTrue(outer_pu__.equals(df3[view]))
