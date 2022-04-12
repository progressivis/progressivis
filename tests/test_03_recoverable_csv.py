from __future__ import annotations

from io import StringIO
import numpy as np
import pandas as pd
from . import ProgressiveTest
from progressivis import Print
from progressivis.core import aio, Sink
from progressivis.io import SimpleCSVLoader

from typing import Optional, Any, Tuple, Callable


def make_num_csv(
    n_rows: int,
    n_cols: int,
    rand_func: Callable,
    intruder: Optional[Tuple[int, int, Any]] = None,
) -> StringIO:
    """
    intruder: Tuple(row, col, value)
    """
    size = n_rows * n_cols
    iarr = rand_func(0, 10_000, size=size)
    str_iarr = [str(i) for i in iarr]
    ord_a = ord("A")
    colnames = [chr(i) for i in range(ord_a, ord_a + n_cols)]
    mx = np.array(colnames + str_iarr, dtype=object).reshape(-1, n_cols)
    if intruder:
        ir, ic, iv = intruder
        mx[ir + 1, ic] = iv  # +1 => header offset
    sio = StringIO()
    for i in range(n_rows + 1):
        print(",".join(mx[i, :]), file=sio)
    sio.seek(0)
    return sio


def make_int_csv(
    n_rows: int, n_cols: int, intruder: Optional[Tuple[int, int, Any]] = None
) -> StringIO:
    return make_num_csv(n_rows, n_cols, np.random.randint, intruder)


def make_float_csv(
    n_rows: int, n_cols: int, intruder: Optional[Tuple[int, int, Any]] = None
) -> StringIO:
    return make_num_csv(n_rows, n_cols, np.random.normal, intruder)


class TestProgressiveLoadCSV(ProgressiveTest):
    def test_read_csv(self) -> None:
        s = self.scheduler()
        n_rows = 100_000
        sio = make_int_csv(n_rows=n_rows, n_cols=3)
        module = SimpleCSVLoader(sio, index_col=False, scheduler=s, dtype="int64")
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        aio.run(s.start())
        self.assertEqual(len(module.table), n_rows)

    def _func_read_int_csv_with_intruder(self, dtype, intruder) -> None:
        s = self.scheduler()
        n_rows = 100_000
        i_row = n_rows - 42
        sio = make_int_csv(n_rows=n_rows, n_cols=3, intruder=(i_row, 1, intruder))
        # csv_name = "/tmp/foo.csv"
        # with open(csv_name, "w") as csv:
        #     csv.write(sio.getvalue())
        df = pd.read_csv(sio)
        sio.seek(0)
        df.loc[i_row, "B"] = np.iinfo(dtype).max
        module = SimpleCSVLoader(sio, index_col=False, scheduler=s, dtype=dtype)
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
        self.assertTrue(i_row in anomalies)
        self.assertTrue(anomalies[i_row]["B"] == intruder)  # type:ignore

    def test_read_int_csv_with_intruder_64(self) -> None:
        self._func_read_int_csv_with_intruder(dtype="int64", intruder="Intruder")

    def test_read_int_csv_with_intruder_32(self) -> None:
        self._func_read_int_csv_with_intruder(dtype="int32", intruder="Intruder")

    # def test_read_int_csv_with_intruder_32_overflow(self) -> None:
    #    self._func_read_int_csv_with_intruder(dtype="int32", intruder=str(2**40))

    def _func_read_float_csv_with_intruder(self, na_filter, intruder, dtype) -> None:
        s = self.scheduler()
        n_rows = 100_000
        i_row = n_rows - 42
        sio = make_float_csv(n_rows=n_rows, n_cols=3, intruder=(i_row, 1, intruder))
        df = pd.read_csv(sio)
        sio.seek(0)
        df.loc[i_row, "B"] = np.nan
        module = SimpleCSVLoader(
            sio, index_col=False, scheduler=s, dtype=dtype, na_filter=na_filter
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.anomalies
        pr2 = Print(proc=self.terse, scheduler=s)
        pr2.input[0] = module.output.missing
        aio.run(s.start())
        self.assertEqual(len(module.table), len(df))
        self.assertTrue(
            np.allclose(
                module.table.to_array(), df.values.astype(dtype), equal_nan=True
            )
        )
        anomalies = module.anomalies()
        assert anomalies
        self.assertTrue(i_row in anomalies)
        self.assertTrue(anomalies[i_row]["B"] == intruder)  # type: ignore
        missing = module.missing()
        assert missing
        self.assertTrue(i_row in missing["B"])  # type:ignore

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
