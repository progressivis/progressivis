from __future__ import annotations

from . import ProgressiveTest
from progressivis.core import aio, notNone
from progressivis import Every
from progressivis.io import CSVLoader
from progressivis.stats import RandomPTable
from progressivis.stats import Histogram2D, Min, Max
from progressivis.vis import Heatmap
from progressivis.datasets import get_dataset
from progressivis.table.stirrer import Stirrer, StirrerView
import pandas as pd
import numpy as np
import fast_histogram as fh  # type: ignore

from typing import Any


class TestHistogram2D(ProgressiveTest):
    def test_histogram2d(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        min_ = Min(scheduler=s)
        min_.input[0] = csv.output.result
        max_ = Max(scheduler=s)
        max_.input[0] = csv.output.result
        histogram2d = Histogram2D(
            1, 2, xbins=100, ybins=100, scheduler=s
        )  # columns are called 1..30
        histogram2d.input[0] = csv.output.result
        histogram2d.input.min = min_.output.result
        histogram2d.input.max = max_.output.result
        heatmap = Heatmap(filename="histo_%03d.png", scheduler=s)
        heatmap.input.array = histogram2d.output.result
        pr = Every(proc=self.terse, scheduler=s)
        pr.input[0] = heatmap.output.result
        aio.run(csv.scheduler().start())
        _ = histogram2d.trace_stats()

    def test_histogram2d1(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        min_ = Min(scheduler=s)
        min_.input[0] = csv.output.result
        max_ = Max(scheduler=s)
        max_.input[0] = csv.output.result
        histogram2d = Histogram2D(
            1, 2, xbins=100, ybins=100, scheduler=s
        )  # columns are called 1..30
        histogram2d.input[0] = csv.output.result
        histogram2d.input.min = min_.output.result
        histogram2d.input.max = max_.output.result
        heatmap = Heatmap(filename="histo_%03d.png", scheduler=s)
        heatmap.input.array = histogram2d.output.result
        pr = Every(proc=self.terse, scheduler=s)
        pr.input[0] = heatmap.output.result
        aio.run(csv.scheduler().start())
        assert histogram2d.result is not None
        last = notNone(histogram2d.result.last()).to_dict()
        h1 = last["array"]
        bounds = [[last["ymin"], last["ymax"]], [last["xmin"], last["xmax"]]]
        df = pd.read_csv(
            get_dataset("bigfile"), header=None, usecols=[1, 2]  # type: ignore
        )
        v = df.to_numpy()  # .reshape(-1, 2)
        bins = [histogram2d.params.ybins, histogram2d.params.xbins]
        h2 = fh.histogram2d(v[:, 1], v[:, 0], bins=bins, range=bounds)
        h2 = np.flip(h2, axis=0)  # type: ignore
        self.assertTrue(np.allclose(h1, h2))

    def t_histogram2d_impl(self, **kw: Any) -> None:
        s = self.scheduler()
        random = RandomPTable(3, rows=100000, scheduler=s)
        stirrer = Stirrer(update_column="_2", fixed_step_size=1000, scheduler=s, **kw)
        stirrer.input[0] = random.output.result
        min_ = Min(scheduler=s)
        min_.input[0] = stirrer.output.result
        max_ = Max(scheduler=s)
        max_.input[0] = stirrer.output.result
        histogram2d = Histogram2D(
            0, 1, xbins=100, ybins=100, scheduler=s
        )  # columns are called 1..30
        histogram2d.input[0] = stirrer.output.result
        histogram2d.input.min = min_.output.result
        histogram2d.input.max = max_.output.result
        heatmap = Heatmap(filename="histo_%03d.png", scheduler=s)
        heatmap.input.array = histogram2d.output.result
        pr = Every(proc=self.terse, scheduler=s)
        pr.input[0] = heatmap.output.result
        aio.run(s.start())
        assert histogram2d.result is not None
        assert stirrer.result is not None
        last = notNone(histogram2d.result.last()).to_dict()
        h1 = last["array"]
        bounds = [[last["ymin"], last["ymax"]], [last["xmin"], last["xmax"]]]
        t = stirrer.result.loc[:, ["_1", "_2"]]
        assert t is not None
        v = t.to_array()
        bins = [histogram2d.params.ybins, histogram2d.params.xbins]
        h2 = fh.histogram2d(v[:, 1], v[:, 0], bins=bins, range=bounds)
        h2 = np.flip(h2, axis=0)  # type: ignore
        self.assertEqual(np.sum(h1), np.sum(h2))
        self.assertListEqual(h1.reshape(-1).tolist(), h2.reshape(-1).tolist())

    def test_histogram2d4(self) -> None:
        s = self.scheduler()
        random = RandomPTable(3, rows=100000, scheduler=s)
        stirrer = StirrerView(
            update_column="_2", fixed_step_size=1000, scheduler=s, delete_rows=5
        )
        stirrer.input[0] = random.output.result
        min_ = Min(scheduler=s)
        min_.input[0] = stirrer.output.result
        max_ = Max(scheduler=s)
        max_.input[0] = stirrer.output.result
        histogram2d = Histogram2D(
            0, 1, xbins=100, ybins=100, scheduler=s
        )  # columns are called 1..30
        histogram2d.input[0] = stirrer.output.result
        histogram2d.input.min = min_.output.result
        histogram2d.input.max = max_.output.result
        heatmap = Heatmap(filename="histo_%03d.png", scheduler=s)
        heatmap.input.array = histogram2d.output.result
        pr = Every(proc=self.terse, scheduler=s)
        pr.input[0] = heatmap.output.result
        aio.run(s.start())
        assert histogram2d.result is not None
        assert stirrer.result is not None
        last = notNone(histogram2d.result.last()).to_dict()
        h1 = last["array"]
        bounds = [[last["ymin"], last["ymax"]], [last["xmin"], last["xmax"]]]
        tmp = stirrer.result.loc[:, ["_1", "_2"]]
        assert tmp is not None
        v = tmp.to_array()
        bins = [histogram2d.params.ybins, histogram2d.params.xbins]
        h2 = fh.histogram2d(v[:, 1], v[:, 0], bins=bins, range=bounds)
        h2 = np.flip(h2, axis=0)  # type: ignore
        self.assertEqual(np.sum(h1), np.sum(h2))
        self.assertListEqual(h1.reshape(-1).tolist(), h2.reshape(-1).tolist())

    def test_histogram2d2(self) -> None:
        return self.t_histogram2d_impl(delete_rows=5)

    def test_histogram2d3(self) -> None:
        return self.t_histogram2d_impl(update_rows=5)


if __name__ == "__main__":
    ProgressiveTest.main()
