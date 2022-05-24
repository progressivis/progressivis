from __future__ import annotations

from io import BytesIO
import numpy as np
import pandas as pd
from . import ProgressiveTest
from progressivis import Print
from progressivis.core import aio, Sink
from progressivis.io import PACSVLoader
from pyarrow.csv import ConvertOptions, ReadOptions
from typing import Optional, Any, Tuple, Callable, List


def make_num_csv(
    n_rows: int,
    n_cols: int,
    rand_func: Callable,
    intruders: Optional[List[Tuple[int, int, Any]]] = None,
    artifact: str = "",
) -> BytesIO:
    """
    intruders: List[Tuple(row, col, value)]
    """
    size = n_rows * n_cols
    np.random.seed(42)
    iarr = rand_func(0, 10_000, size=size)
    str_iarr = [str(i) for i in iarr]
    ord_a = ord("A")
    colnames = [f"{chr(i)}{artifact}" for i in range(ord_a, ord_a + n_cols)]
    mx = np.array(colnames + str_iarr, dtype=object).reshape(-1, n_cols)
    if intruders:
        for intruder in intruders:
            ir, ic, iv = intruder
            mx[ir + 1, ic] = iv  # +1 => header offset
    bio = BytesIO()
    for i in range(n_rows + 1):
        bio.write(",".join(mx[i, :]).encode())
        bio.write(b"\n")
    bio.seek(0)
    return bio


def make_int_csv(
    n_rows: int,
    n_cols: int,
    intruders: Optional[List[Tuple[int, int, Any]]] = None,
    artifact: str = "",
) -> BytesIO:
    return make_num_csv(n_rows, n_cols, np.random.randint, intruders, artifact=artifact)


def make_float_csv(
    n_rows: int, n_cols: int, intruders: Optional[List[Tuple[int, int, Any]]] = None
) -> BytesIO:
    return make_num_csv(n_rows, n_cols, np.random.normal, intruders)


class TestProgressiveLoadCSV(ProgressiveTest):
    def _read_csv(self, artifact: str = "", force_valid_ids: bool = True) -> None:
        s = self.scheduler()
        n_rows = 100_000
        bio = make_int_csv(n_rows=n_rows, n_cols=3, artifact=artifact)
        module = PACSVLoader(
            bio, index_col=False, scheduler=s, force_valid_ids=force_valid_ids
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.table), n_rows)

    def test_read_csv(self) -> None:
        self._read_csv()

    def test_read_csv_artifact_fail(self, artifact="?") -> None:
        with self.assertRaises(RuntimeError):
            self._read_csv(artifact=artifact, force_valid_ids=False)

    def test_read_csv_artifact(self, artifact="?") -> None:
        self._read_csv(artifact=artifact)

    def _func_read_int_csv_with_intruder(
        self, dtype, intruder, fixed_step_size=0
    ) -> None:
        s = self.scheduler()
        n_rows = 100_000
        rows = set(np.random.randint(10_000, n_rows - 1, size=1000))
        if intruder:
            intruders = [(r, 1, intruder + str(r)) for r in rows]
        else:
            intruders = [(r, 1, intruder) for r in rows]
        bio = make_int_csv(n_rows=n_rows, n_cols=3, intruders=intruders)
        df = pd.read_csv(bio)
        bio.seek(0)
        df = df.drop(rows)
        cvopts = ConvertOptions(column_types={c: dtype for c in df.columns})
        ropts = ReadOptions(block_size=100_000)
        module = PACSVLoader(
            bio,
            scheduler=s,
            read_options=ropts,
            convert_options=cvopts,
        )
        if fixed_step_size:
            setattr(module, "predict_step_size", lambda x: fixed_step_size)
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.anomalies
        aio.run(s.start())
        self.assertEqual(len(module.table), len(df))
        self.assertTrue(
            np.array_equal(module.table.to_array(), df.values.astype(dtype))
        )
        anomalies = module.anomalies()
        assert anomalies
        self.assertEqual(anomalies["skipped_cnt"], len(rows))
        self.assertEqual(len(anomalies["invalid_values"]), len(rows) if intruder else 0)

    def test_read_int_csv_with_na_64(self) -> None:
        self._func_read_int_csv_with_intruder(dtype="int64", intruder="")

    def test_read_int_csv_with_intruder_64(self) -> None:
        self._func_read_int_csv_with_intruder(dtype="int64", intruder="Intruder")

    def test_read_int_csv_with_na_32(self) -> None:
        self._func_read_int_csv_with_intruder(dtype="int32", intruder="")

    def test_read_int_csv_with_intruder_32(self) -> None:
        self._func_read_int_csv_with_intruder(dtype="int32", intruder="Intruder")

    def _func_read_float_csv_with_intruder(
        self,
        na_filter,
        dtype,
        intruder,
        imputer=None,
    ) -> None:
        s = self.scheduler()
        n_rows = 100_000
        rows = set(np.random.randint(10_000, n_rows - 1, size=1000))
        if intruder:
            intruders = [(r, 1, intruder + str(r)) for r in rows]
        else:
            intruders = [(r, 1, intruder) for r in rows]
        bio = make_float_csv(n_rows=n_rows, n_cols=3, intruders=intruders)
        df = pd.read_csv(bio)
        bio.seek(0)
        df = df.drop(rows)
        cvopts = ConvertOptions(column_types={c: dtype for c in df.columns})
        ropts = ReadOptions(block_size=100_000, use_threads=False)
        module = PACSVLoader(
            bio,
            scheduler=s,
            read_options=ropts,
            convert_options=cvopts,
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.anomalies
        aio.run(s.start())
        self.assertEqual(len(module.table), len(df))
        self.assertTrue(
            np.allclose(
                module.table.to_array(), df.values.astype(dtype), equal_nan=True
            )
        )
        anomalies = module.anomalies()
        assert anomalies
        self.assertEqual(anomalies["skipped_cnt"], len(rows))
        self.assertEqual(len(anomalies["invalid_values"]), len(rows) if intruder else 0)

    def test_read_float_csv_with_intruder_not_na_64(self) -> None:
        self._func_read_float_csv_with_intruder(
            na_filter=False, intruder="", dtype="float64"
        )

    def test_read_float_csv_with_intruder_na_64(self) -> None:
        self._func_read_float_csv_with_intruder(
            na_filter=True, intruder="Intruder", dtype="float64"
        )

    def test_read_float_csv_with_intruder_not_na_32(self) -> None:
        self._func_read_float_csv_with_intruder(
            na_filter=False, intruder="", dtype="float32"
        )

    def test_read_float_csv_with_intruder_na_32(self) -> None:
        self._func_read_float_csv_with_intruder(
            na_filter=True, intruder="Intruder", dtype="float32"
        )


if __name__ == "__main__":
    ProgressiveTest.main()
