from __future__ import annotations

from . import ProgressiveTest
from progressivis import Sink, RandomPTable
from progressivis.core import aio

from progressivis.table.binning_index_nd import BinningIndexND

import numpy as np
from typing import Any


def random_func_outl(size: int, weight: float = 0.01, mul: int = 10, side: int = -1) -> np.ndarray[Any, Any]:
    ol_size = int(size * weight)
    main_size = size - ol_size
    vect_main = np.random.rand(main_size)
    vect_ol = np.random.rand(ol_size)
    return np.hstack([vect_main, vect_ol * mul * side])


outliers_left = random_func_outl


def outliers_right(size: int) -> np.ndarray[Any, Any]:
    return random_func_outl(size, side=1)


def bin_char(v: Any) -> str:
    return "n" if v is None else "."


class TestBinningIndex(ProgressiveTest):
    def check_bins(self, impl, col) -> None:  # type: ignore
        binvect = impl.binvect
        binvect_map = impl.binvect_map
        bin_w = impl.bin_w
        origin = impl.origin
        bins_count = sum([len(binvect[i]) for i in binvect_map])
        self.assertEqual(bins_count, len(col))
        for i in binvect_map:
            values = col.loc[binvect[i]]
            self.assertTrue(np.all(values >= origin + i * bin_w))
            self.assertTrue(np.all(values < origin + (i + 1) * bin_w))

    def test_normal(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=100_000, scheduler=s)
        bin_index = BinningIndexND(scheduler=s)
        bin_index.input[0] = random.output.result["_1", "_2"]
        sink = Sink(scheduler=s)
        sink.input.inp = bin_index.output.result
        aio.run(s.start())
        assert bin_index._impl is not None
        assert "_1" in bin_index._impl
        assert "_2" in bin_index._impl
        for col, impl in bin_index._impl.items():
            print(col, ":", len(impl.binvect))
            print("[")
            for bin in impl.binvect:
                print(bin_char(bin), end="")
            print("]")
        assert random.result
        self.check_bins(bin_index._impl["_1"], random.result["_1"])
        self.check_bins(bin_index._impl["_2"], random.result["_2"])

    def test_outliers_left(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, random=outliers_left, rows=100_000, scheduler=s)
        bin_index = BinningIndexND(scheduler=s)
        bin_index.input[0] = random.output.result["_1", "_2"]
        sink = Sink(scheduler=s)
        sink.input.inp = bin_index.output.result
        aio.run(s.start())
        assert bin_index._impl is not None
        assert "_1" in bin_index._impl
        assert "_2" in bin_index._impl
        for col, impl in bin_index._impl.items():
            print(col, ":", len(impl.binvect))
            print("[")
            for bin in impl.binvect:
                print(bin_char(bin), end="")
            print("]")
        assert random.result
        self.check_bins(bin_index._impl["_1"], random.result["_1"])
        self.check_bins(bin_index._impl["_2"], random.result["_2"])

    def test_outliers_right(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, random=outliers_right, rows=100_000, scheduler=s)
        bin_index = BinningIndexND(scheduler=s)
        bin_index.input[0] = random.output.result["_1", "_2"]
        sink = Sink(scheduler=s)
        sink.input.inp = bin_index.output.result
        aio.run(s.start())
        assert bin_index._impl is not None
        assert "_1" in bin_index._impl
        assert "_2" in bin_index._impl
        for col, impl in bin_index._impl.items():
            print(col, ":", len(impl.binvect))
            print("[")
            for bin in impl.binvect:
                print(bin_char(bin), end="")
            print("]")
        assert random.result
        self.check_bins(bin_index._impl["_1"], random.result["_1"])
        self.check_bins(bin_index._impl["_2"], random.result["_2"])
