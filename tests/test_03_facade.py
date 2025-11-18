"Test for SlotHub"
from . import ProgressiveTest

from progressivis import notNone, Sink, Tick, Min, Max, RandomPTable, Histogram1D, Histogram2D
from progressivis.core import aio
from progressivis.core.module_facade import ModuleFacade
from progressivis.table.table_facade import TableFacade
import numpy as np

from typing import Any, Dict, cast


class TestModuleFacade(ProgressiveTest):
    def test_module_facade(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=10000, scheduler=s)
        min_ = Min(name="min_" + str(hash(random)), scheduler=s)
        min_.input[0] = random.output.result
        max_ = Max(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = random.output.result
        hub = ModuleFacade()
        hub.add_proxy("min", "result", min_)
        hub.add_proxy("max", "result", max_)
        sink_min = Sink(scheduler=s)
        sink_min.input[0] = hub.output.min
        sink_max = Sink(scheduler=s)
        sink_max.input[0] = hub.output.max
        aio.run(s.start())
        assert random.result is not None
        assert min_.result is not None
        assert max_.result is not None
        res1 = random.result.min()
        res2 = min_.result
        self.compare(res1, res2)
        res1 = random.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_table_module(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=10000, scheduler=s)
        tabmod = TableFacade.get_or_create(random, "result")
        sink_min = Sink(scheduler=s)
        sink_min.input[0] = tabmod.output.min
        sink_max = Sink(scheduler=s)
        sink_max.input[0] = tabmod.output.max
        aio.run(s.start())

    def test_table_module_child(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=10000, scheduler=s)
        tabmod = TableFacade.get_or_create(random, "result")
        sink_min = Sink(scheduler=s)
        sink_min.input[0] = tabmod.child.min.output.result
        sink_max = Sink(scheduler=s)
        sink_max.input[0] = tabmod.child.max.output.result
        aio.run(s.start())

    def test_sample(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=10000, scheduler=s)
        tabmod = TableFacade.get_or_create(random, "result")
        prt = Tick(scheduler=s)
        tabmod.child.sample.params.samples = 10
        prt.input[0] = tabmod.child.sample.output.result
        prt2 = Tick(scheduler=s)
        prt2.input[0] = tabmod.child.sample.output.select
        aio.run(s.start())
        assert hasattr(tabmod.child.sample, "result")
        self.assertEqual(len(tabmod.child.sample.result), 10)
        assert hasattr(tabmod.child.sample, "pintset")
        self.assertEqual(len(tabmod.child.sample.pintset), 10)

    def test_table_module_cols(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=10000, scheduler=s)
        tabmod = TableFacade.get_or_create(random, "result")
        min_ = Min(name="min_" + str(hash(random)), scheduler=s)
        min_.input.table = tabmod.output.main["_1", "_2", "_3"]
        pr = Tick(scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        assert tabmod.output.main is not None
        assert min_.result is not None
        assert hasattr(tabmod.module, "result")
        res1 = tabmod.module.result.loc[:, ["_1", "_2", "_3"]].min()
        res2 = min_.result
        self.compare(res1, res2)

    def test_table_module_log(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=10000, scheduler=s)
        tabmod = TableFacade.get_or_create(random, "result")
        min_ = Min(name="min_" + str(hash(random)), scheduler=s)
        min_.input.table = tabmod.output.log["_1", "_2", "_3"]
        pr = Tick(scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        assert tabmod.output.log is not None
        assert min_.result is not None
        assert hasattr(tabmod.module, "result")
        res1 = np.log(tabmod.module.result.loc[:, ["_1", "_2", "_3"]].to_array()).min(axis=0)
        res2 = min_.result
        self.assertTrue(np.allclose(res1, list(res2.values())))

    def test_table_module_configure(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=10000, scheduler=s)
        tabmod = TableFacade.get_or_create(random, "result")
        tabmod.configure(base="min", hints=["_1", "_2", "_3"], name="min3")
        sink = Sink(scheduler=s)
        sink.input[0] = tabmod.output.min3
        aio.run(s.start())
        assert hasattr(tabmod.child.main, "result")
        assert hasattr(tabmod.child.min3, "result")
        assert len(tabmod.child.min3.result) == 3
        res1 = tabmod.child.main.result.loc[:, ["_1", "_2", "_3"]].min()
        res2 = tabmod.child.min3.result
        self.compare(res1, res2)

    def test_table_module_configure_hist(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=10000, scheduler=s)
        tabmod = TableFacade.get_or_create(random, "result")
        tabmod.configure(base="min", hints=["_1"], name="min_1")
        tabmod.configure(base="max", hints=["_1"], name="max_1")
        tabmod.configure(base="histogram", hints=["_1",], name="hist_1",
                         connect=dict(min="min_1", max="max_1"))
        sink = Sink(scheduler=s)
        sink.input[0] = tabmod.output.hist_1
        aio.run(s.start())
        histogram1d = cast(Histogram1D, notNone(tabmod.get('hist_1')).output_module)
        last = notNone(histogram1d.result)
        h1 = last["array"]
        bounds = (last["min"], last["max"])
        tab = notNone(random.result).loc[:, ["_1"]]
        assert tab is not None
        v = tab.to_array().reshape(-1)
        h2, _ = np.histogram(
            v, bins=histogram1d.params.bins, density=False, range=bounds
        )
        self.assertEqual(np.sum(h1), np.sum(h2))
        self.assertListEqual(h1.tolist(), h2.tolist())

    def test_table_module_configure_hist2d(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=10000, scheduler=s)
        tabmod = TableFacade.get_or_create(random, "result")
        tabmod.configure(base="min", hints=["_1", "_2"], name="min_1_2")
        tabmod.configure(base="max", hints=["_1", "_2"], name="max_1_2")
        tabmod.configure(base="histogram2d", hints=dict(x="_1", y="_2"), name="hist2d_1_2",
                         connect=dict(min="min_1_2", max="max_1_2"))
        sink = Sink(scheduler=s)
        sink.input[0] = tabmod.output.hist2d_1_2
        aio.run(s.start())
        histogram2d = cast(Histogram2D, notNone(tabmod.get('hist2d_1_2')).output_module)
        last = notNone(notNone(histogram2d.result).last()).to_dict()
        h1 = last["array"]
        bounds = [[last["ymin"], last["ymax"]], [last["xmin"], last["xmax"]]]
        tab = notNone(random.result).loc[:, ["_1", "_2"]]
        assert tab is not None
        v = tab.to_array()
        bins = [histogram2d.params.ybins, histogram2d.params.xbins]
        h2 = np.histogram2d(v[:, 1], v[:, 0], bins=bins, range=bounds)[0]
        h2 = np.flip(h2, axis=0)
        self.assertTrue(np.allclose(h1, h2))

    def compare(self, res1: Dict[str, Any], res2: Dict[str, Any]) -> None:
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        self.assertTrue(np.allclose(v1, v2))


if __name__ == "__main__":
    ProgressiveTest.main()
