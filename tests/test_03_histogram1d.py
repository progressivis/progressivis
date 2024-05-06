from . import ProgressiveTest
from progressivis.core import aio
from progressivis import Every
from progressivis.io import CSVLoader
from progressivis.stats import Histogram1D, Min, Max
from progressivis.datasets import get_dataset
from progressivis.table.stirrer import Stirrer
import pandas as pd
import numpy as np

from typing import Any


class TestHistogram1D(ProgressiveTest):
    def test_histogram1d(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        min_ = Min(scheduler=s)
        min_.input[0] = csv.output.result
        max_ = Max(scheduler=s)
        max_.input[0] = csv.output.result
        histogram1d = Histogram1D("_2", scheduler=s)  # columns are called 1..30
        histogram1d.input[0] = csv.output.result
        histogram1d.input.min = min_.output.result
        histogram1d.input.max = max_.output.result
        pr = Every(proc=self.terse, scheduler=s)
        pr.input[0] = histogram1d.output.result
        aio.run(s.start())
        _ = histogram1d.trace_stats()

    def test_histogram1d1(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        min_ = Min(scheduler=s)
        min_.input[0] = csv.output.result
        max_ = Max(scheduler=s)
        max_.input[0] = csv.output.result
        histogram1d = Histogram1D("_2", scheduler=s)  # columns are called 1..30
        histogram1d.input[0] = csv.output.result
        histogram1d.input.min = min_.output.result
        histogram1d.input.max = max_.output.result
        pr = Every(proc=self.terse, scheduler=s)
        pr.input[0] = histogram1d.output.result
        aio.run(s.start())
        _ = histogram1d.trace_stats()
        assert histogram1d.result is not None
        last = histogram1d.result
        h1 = last["array"]
        bounds = (last["min"], last["max"])
        df = pd.read_csv(
            get_dataset("bigfile"), header=None, usecols=[2]
        )
        v = df.to_numpy().reshape(-1)
        h2, _ = np.histogram(
            v, bins=histogram1d.params.bins, density=False, range=bounds
        )
        self.assertListEqual(h1.tolist(), h2.tolist())

    def t_histogram1d_impl(self, **kw: Any) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        stirrer = Stirrer(update_column="_2", fixed_step_size=1000, scheduler=s, **kw)
        stirrer.input[0] = csv.output.result
        min_ = Min(scheduler=s)
        min_.input[0] = stirrer.output.result
        max_ = Max(scheduler=s)
        max_.input[0] = stirrer.output.result
        histogram1d = Histogram1D("_2", scheduler=s)  # columns are called 1..30
        histogram1d.input[0] = stirrer.output.result
        histogram1d.input.min = min_.output.result
        histogram1d.input.max = max_.output.result

        # pr = Print(scheduler=s)
        pr = Every(proc=self.terse, scheduler=s)
        pr.input[0] = histogram1d.output.result
        aio.run(s.start())
        assert histogram1d.result is not None
        assert stirrer.result is not None
        _ = histogram1d.trace_stats()
        last = histogram1d.result
        h1 = last["array"]
        bounds = (last["min"], last["max"])
        tab = stirrer.result.loc[:, ["_2"]]
        assert tab is not None
        v = tab.to_array().reshape(-1)
        h2, _ = np.histogram(
            v, bins=histogram1d.params.bins, density=False, range=bounds
        )
        self.assertEqual(np.sum(h1), np.sum(h2))
        self.assertListEqual(h1.tolist(), h2.tolist())

    def test_histogram1d2(self) -> None:
        return self.t_histogram1d_impl(delete_rows=5)

    def test_histogram1d3(self) -> None:
        return self.t_histogram1d_impl(update_rows=5)


if __name__ == "__main__":
    ProgressiveTest.main()
