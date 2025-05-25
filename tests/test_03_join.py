from __future__ import annotations

import os
from . import ProgressiveTest, skipIf
from progressivis.core import aio
from progressivis import (
    Sink, ParquetLoader, SimpleCSVLoader, Join, get_dataset
)

import pandas as pd
import numpy as np
from itertools import product
from io import StringIO
from typing import Any, Sequence, Tuple, List, cast

# PARQUET_FILE = "nyc-taxi/newstyle_500k_yellow_tripdata_2015-01.parquet"
PARQUET_FILE = get_dataset("newshort-taxis2015-01_parquet")
# CSV_URL = "https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv"
CSV_URL = "https://www.aviz.fr/nyc-taxi/taxi-zone-lookup.csv.bz2"
# CSV_URL = "../nyc-taxi/taxi+_zone_lookup.csv"
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
    TAXIS = pd.read_parquet(PARQUET_FILE, columns=TAXI_COLS)
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


FK2_N = 10


def np_array_equal(arr1: Any, arr2: Any) -> bool:
    return np.array_equal(arr1, arr2)


def generate_random_csv_left(
    rows: int = 300_000, seed: int = 42, choice: Sequence[str] = ("A", "B", "C", "D"),
) -> Tuple[pd.DataFrame, str]:
    np.random.seed(seed)
    df = pd.DataFrame(
        {
            "card_left": range(rows),
            "I": np.random.randint(0, 10_000, size=rows, dtype=int),
            "J": np.random.randint(0, 15_000, size=rows, dtype=int),
            "FK_1": np.random.choice(choice, rows),
            "FK_2": np.random.randint(0, FK2_N, size=rows, dtype=int),
        }
    )
    sio = StringIO()
    df.to_csv(sio, index=False)
    sio.seek(0)
    return df, sio.getvalue()


def generate_random_csv_right(
        seq1: Tuple[Any, ...] = ("A", "B", "C", "D"),
        seq2: Sequence[Any] = range(FK2_N)
) -> Tuple[pd.DataFrame, str]:
    pk1, pk2 = list(zip(*product(seq1, seq2)))
    info = [f"{tpl[0]}{tpl[1]}" for tpl in product(seq1, seq2)]
    df = pd.DataFrame(
        {
            "PK_1": pk1[3:] + ("X", "Y", "Z"),
            "PK_2": pk2[3:] + (-1, -6, -15),
            "info": info[3:] + ["X_1", "Y_6", "Z_15"],
            "card_right": range(len(info)),
        }
    )
    sio = StringIO()
    df.to_csv(sio, index=False)
    sio.seek(0)
    return df, sio.getvalue()


df_right, csv_right = generate_random_csv_right()
df_left, csv_left = generate_random_csv_left()
df_inner = df_left.join(
    df_right.set_index(["PK_1", "PK_2"]), on=["FK_1", "FK_2"], how="inner"
)
df_left_outer = df_left.join(
    df_right.set_index(["PK_1", "PK_2"]), on=["FK_1", "FK_2"], how="left"
)
df_outer = df_left.join(
    df_right.set_index(["PK_1", "PK_2"]), on=["FK_1", "FK_2"], how="outer"
)


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
            related_on=["DOLocationID"],
            primary_on=["LocationID"],
            related_cols=TAXI_COLS,
        )
        sink = Sink(scheduler=s)
        sink.input.inp = join.output.result
        aio.run(s.start())
        assert join.result is not None
        df = join.result.to_df(
            to_datetime=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        )
        self.assertEqual(len(df), len(INNER))
        self.assertEqual(
            set(df.columns) | {"control_id", "lookup_id"}, set(INNER.columns)
        )
        for col in df.columns:
            self.assertTrue(np_array_equal(df[col].values, INNER[col].values))

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
        assert join.result is not None
        df = join.result.to_df(
            to_datetime=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        )
        self.assertEqual(len(df), len(LEFT_OUTER))
        self.assertEqual(
            set(df.columns) | {"control_id", "lookup_id"}, set(LEFT_OUTER.columns)
        )
        for col in df.columns:
            self.assertTrue(np_array_equal(df[col].values, LEFT_OUTER[col].values))
        assert join._primary_outer is not None
        df2 = join._primary_outer.to_df().rename(columns={"LocationID": "DOLocationID"})
        df_concat = (
            pd.concat([df[df2.columns], df2])
            .set_index("DOLocationID")
            .sort_values("DOLocationID")
        )
        for col in df_concat.columns:
            self.assertTrue(np_array_equal(df_concat[col].values, OUTER[col].values))

    #  @skipIf(True, "Too long")
    def test_inner_pu(self) -> None:
        s = self.scheduler()
        parquet = ParquetLoader(PARQUET_FILE, columns=TAXI_COLS, scheduler=s,)
        csv = SimpleCSVLoader(CSV_URL, skiprows=LOOKUP_SKIP_ROWS, scheduler=s,)
        self.assertTrue(parquet.result is None)
        join = Join(how="inner", scheduler=s)
        join.create_dependent_modules(
            related_module=parquet,
            primary_module=csv,
            related_on=["DOLocationID"],
            primary_on=["LocationID"],
            related_cols=TAXI_COLS,
        )
        join_pu = Join(how="inner", scheduler=s)
        join_pu.create_dependent_modules(
            related_module=join,
            primary_module=csv,
            related_on=["PULocationID"],
            primary_on=["LocationID"],
            related_cols=TAXI_COLS + list(LOOKUP.columns),
            suffix="_pu",
        )
        sink = Sink(scheduler=s)
        sink.input.inp = join_pu.output.result
        aio.run(s.start())
        assert join_pu.result is not None
        df = join_pu.result.to_df(
            to_datetime=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        )
        self.assertEqual(len(df), len(INNER_PU))
        self.assertEqual(
            set(df.columns) | {"control_id", "lookup_id", "lookup_id_pu"},
            set(INNER_PU.columns),
        )
        for col in df.columns:
            self.assertTrue(np_array_equal(df[col].values, INNER_PU[col].values))

    # @skipIf(True, "Too long")
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
        assert join_pu.result is not None
        df = join_pu.result.to_df(
            to_datetime=["tpep_pickup_datetime", "tpep_dropoff_datetime"]
        )
        self.assertEqual(len(df), len(LEFT_OUTER_PU))
        self.assertEqual(
            set(df.columns) | {"control_id", "lookup_id", "lookup_id_pu"},
            set(LEFT_OUTER_PU.columns),
        )
        for col in df.columns:
            self.assertTrue(np_array_equal(df[col].values, LEFT_OUTER_PU[col].values))
        assert join._primary_outer is not None
        df2 = join._primary_outer.to_df().rename(columns={"LocationID": "PULocationID"})
        assert join_pu._primary_outer is not None
        df3 = join_pu._primary_outer.to_df().rename(
            columns={"LocationID": "PULocationID"}
        )
        view: List[str] = list(df2.columns)[1:]  # i.e. 'Borough', 'Zone', 'service_zone'
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
        outer_pu__.columns = cast(pd.Index[Any], view)
        self.assertTrue(outer_pu__.equals(df3[view]))


class TestProgressiveJoin2(ProgressiveTest):
    def test_inner(self) -> None:
        s = self.scheduler()
        sio_left = StringIO(csv_left)
        sio_left.seek(0)
        related = SimpleCSVLoader(sio_left, scheduler=s,)
        sio_right = StringIO(csv_right)
        sio_right.seek(0)
        primary = SimpleCSVLoader(sio_right, scheduler=s,)
        self.assertTrue(related.result is None)
        join = Join(how="inner", scheduler=s)
        join.create_dependent_modules(
            related_module=related,
            primary_module=primary,
            related_on=["FK_1", "FK_2"],
            primary_on=["PK_1", "PK_2"],
            related_cols=cast(List[str], df_left.columns),
        )
        sink = Sink(scheduler=s)
        sink.input.inp = join.output.result
        aio.run(s.start())
        assert join.result is not None
        df = join.result.to_df()
        self.assertEqual(len(df), len(df_inner))
        self.assertEqual(set(df.columns), set(df_inner.columns))
        sorted_inner = df_inner.sort_values("card_left")
        for col in df.columns:
            self.assertTrue(np_array_equal(df[col].values, sorted_inner[col].values))

    def test_outer(self) -> None:
        s = self.scheduler()
        sio_left = StringIO(csv_left)
        sio_left.seek(0)
        related = SimpleCSVLoader(sio_left, scheduler=s,)
        sio_right = StringIO(csv_right)
        sio_right.seek(0)
        primary = SimpleCSVLoader(sio_right, scheduler=s,)
        self.assertTrue(related.result is None)
        join = Join(how="outer", fillna=0, scheduler=s)
        join.create_dependent_modules(
            related_module=related,
            primary_module=primary,
            related_on=["FK_1", "FK_2"],
            primary_on=["PK_1", "PK_2"],
            related_cols=cast(List[str], df_left.columns),
        )
        sink = Sink(scheduler=s)
        sink.input.inp = join.output.result
        sink2 = Sink(scheduler=s)
        sink2.input.inp = join.output.primary_outer
        aio.run(s.start())
        assert join.result is not None
        df = join.result.to_df()
        self.assertEqual(len(df), len(df_left_outer))
        self.assertEqual(set(df.columns), set(df_left_outer.columns))
        sorted_left_outer = df_left_outer.sort_values("card_left").fillna(0)
        for col in df.columns:
            self.assertTrue(
                np_array_equal(df[col].values, sorted_left_outer[col].values)
            )
        assert join._primary_outer is not None
        df2 = join._primary_outer.to_df().rename(
            columns={"PK_1": "FK_1", "PK_2": "FK_2"}
        )
        df_concat = (
            pd.concat([df[df2.columns], df2])
            .set_index(["FK_1", "FK_2"])
            .sort_values(["FK_1", "FK_2"])
        )
        sorted_outer = df_outer.sort_values(["FK_1", "FK_2"]).fillna(0)
        for col in df_concat.columns:
            self.assertTrue(
                np_array_equal(df_concat[col].values, sorted_outer[col].values)
            )
