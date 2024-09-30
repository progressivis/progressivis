from __future__ import annotations

from io import StringIO
import os
import numpy as np
import pandas as pd
from . import ProgressiveTest, skipIf
from progressivis import Print
from progressivis.core.api import Sink
from progressivis.core import aio

from progressivis.io import SimpleCSVLoader
from progressivis.stats.utils import SimpleImputer

from typing import Optional, Any, Tuple, Callable, Literal, Union

NumType = Union[
    Literal["int32"], Literal["int64"], Literal["float32"], Literal["float64"]
]


def make_num_csv(
    n_rows: int,
    n_cols: int,
    rand_func: Callable[..., Any],
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


@skipIf(os.getenv("CI"), "disabled on CI => to be improved")
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
        assert module.result is not None
        self.assertEqual(len(module.result), n_rows)

    def _func_read_int_csv_with_intruder(
        self,
        dtype: NumType,
        intruder: str,
        imputer: Optional[SimpleImputer] = None,
        atol: float = 0,
        fixed_step_size: int = 0,
    ) -> None:
        s = self.scheduler()
        n_rows = 100_000
        i_row = n_rows - 42
        sio = make_int_csv(n_rows=n_rows, n_cols=3, intruder=(i_row, 1, intruder))
        df = pd.read_csv(sio)
        sio.seek(0)

        def _subst() -> Any:
            if not imputer:
                return np.iinfo(dtype).max
            strategy = imputer.get_strategy("B")
            if strategy == "mean":
                return df["B"].drop(i_row).astype(dtype).mean()
            if strategy == "median":
                return df["B"].drop(i_row).astype(dtype).median()
            if strategy == "constant":
                return 55
            if strategy == "most_frequent":
                assert fixed_step_size
                return df.loc[: n_rows - fixed_step_size, "B"].astype(dtype).mode()[0]

        df.loc[i_row, "B"] = _subst()
        module = SimpleCSVLoader(
            sio, index_col=False, scheduler=s, dtype=dtype, imputer=imputer
        )
        if fixed_step_size:
            setattr(module, "predict_step_size", lambda x: fixed_step_size)
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.anomalies
        aio.run(s.start())
        assert module.result is not None
        self.assertEqual(len(module.result), len(df))
        if imputer:
            self.assertTrue(
                abs(
                    module.result.to_array()[i_row, 1]
                    - df.values.astype(dtype)[i_row, 1]
                )
                <= atol
            )
        else:
            self.assertTrue(
                np.array_equal(module.result.to_array(), df.values.astype(dtype))
            )
        anomalies = module.anomalies
        assert anomalies
        # actualy there are int keys in anomalies ...
        self.assertTrue(i_row in anomalies)  # type:ignore
        self.assertTrue(anomalies[i_row]["B"] == intruder)  # type:ignore

    def test_read_int_csv_with_intruder_64(self) -> None:
        self._func_read_int_csv_with_intruder(dtype="int64", intruder="Intruder")

    def test_read_int_csv_with_intruder_32(self) -> None:
        self._func_read_int_csv_with_intruder(dtype="int32", intruder="Intruder")

    def test_read_int_csv_with_intruder_64_default(self) -> None:
        self._func_read_int_csv_with_intruder(
            dtype="int64", intruder="Intruder", imputer=SimpleImputer(), atol=20
        )

    def test_read_int_csv_with_intruder_64_median(self) -> None:
        self._func_read_int_csv_with_intruder(
            dtype="int64",
            intruder="Intruder",
            imputer=SimpleImputer("median"),
            atol=100,
        )

    def test_read_int_csv_with_intruder_64_freq(self) -> None:
        self._func_read_int_csv_with_intruder(
            dtype="int64",
            intruder="Intruder",
            imputer=SimpleImputer("most_frequent"),
            fixed_step_size=10_000,
        )

    def test_read_int_csv_with_intruder_64_constant(self) -> None:
        self._func_read_int_csv_with_intruder(
            dtype="int64",
            intruder="Intruder",
            imputer=SimpleImputer("constant", fill_values=55),
            fixed_step_size=10_000,
        )

    def _func_read_float_csv_with_intruder(
        self,
        na_filter: bool,
        dtype: NumType,
        intruder: str,
        imputer: Optional[SimpleImputer] = None,
        atol: float = 0,
    ) -> None:
        s = self.scheduler()
        n_rows = 100_000
        i_row = n_rows - 42
        sio = make_float_csv(n_rows=n_rows, n_cols=3, intruder=(i_row, 1, intruder))
        df = pd.read_csv(sio)
        sio.seek(0)

        def _subst() -> Any:
            if not imputer:
                return np.nan
            strategy = imputer.get_strategy("B")
            if strategy == "mean":
                return df["B"].drop(i_row).astype(dtype).mean()
            if strategy == "median":
                return df["B"].drop(i_row).astype(dtype).median()
            if strategy == "constant":
                return 55.0
            # if strategy == "most_frequent":
            #    assert fixed_step_size
            #    return df.loc[: n_rows - fixed_step_size, "B"].astype(dtype).mode()[0]

        df.loc[i_row, "B"] = _subst()
        module = SimpleCSVLoader(
            sio,
            index_col=False,
            scheduler=s,
            dtype=dtype,
            na_filter=na_filter,
            imputer=imputer,
        )
        self.assertTrue(module.result is None)
        sink = Sink(scheduler=s)
        sink.input.inp = module.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = module.output.anomalies
        pr2 = Print(proc=self.terse, scheduler=s)
        pr2.input[0] = module.output.missing
        aio.run(s.start())
        assert module.result is not None
        self.assertEqual(len(module.result), len(df))
        if imputer:
            print(module.result.to_array()[i_row, 1], df.values.astype(dtype)[i_row, 1])
            self.assertTrue(
                abs(
                    module.result.to_array()[i_row, 1]
                    - df.values.astype(dtype)[i_row, 1]
                )
                <= atol
            )
        else:
            self.assertTrue(
                np.allclose(
                    module.result.to_array(), df.values.astype(dtype), equal_nan=True
                )
            )

        anomalies = module.anomalies
        assert anomalies
        self.assertTrue(i_row in anomalies)  # type: ignore
        self.assertTrue(anomalies[i_row]["B"] == intruder)  # type: ignore
        missing = module.missing
        assert missing
        self.assertTrue(i_row in missing["B"])

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

    def test_read_float_csv_with_intruder_not_na_64_mean(self) -> None:
        self._func_read_float_csv_with_intruder(
            na_filter=False,
            intruder="",
            dtype="float64",
            imputer=SimpleImputer(),
            atol=100,
        )

    def test_read_float_csv_with_intruder_not_na_64_median(self) -> None:
        self._func_read_float_csv_with_intruder(
            na_filter=False,
            intruder="",
            dtype="float64",
            imputer=SimpleImputer(strategy="median"),
            atol=200,
        )

    def test_read_float_csv_with_intruder_not_na_64_constant(self) -> None:
        self._func_read_float_csv_with_intruder(
            na_filter=False,
            intruder="",
            dtype="float64",
            imputer=SimpleImputer(strategy="constant", fill_values=55),
        )


if __name__ == "__main__":
    ProgressiveTest.main()
