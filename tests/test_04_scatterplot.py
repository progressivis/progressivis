from __future__ import annotations

from progressivis import Every, Print, Scheduler
from progressivis.io import CSVLoader
from progressivis.vis import MCScatterPlot
from progressivis.datasets import get_dataset
from progressivis.stats import RandomPTable
from progressivis.core import aio
from . import ProgressiveTest

LOWER_X = 0.2
LOWER_Y = 0.3
UPPER_X = 0.8
UPPER_Y = 0.7


class TestScatterPlot(ProgressiveTest):
    def tearDown(self) -> None:
        TestScatterPlot.cleanup()

    def test_scatterplot(self) -> None:
        s = self.scheduler(clean=True)
        with s:
            csv = CSVLoader(
                get_dataset("smallfile"),
                index_col=False,
                header=None,
                force_valid_ids=True,
                scheduler=s,
            )
            sp = MCScatterPlot(
                scheduler=s, classes=[("Scatterplot", "_1", "_2")], approximate=True
            )
            sp.create_dependent_modules(csv, "result")
            cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
            cnt.input[0] = csv.output.result
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = sp.output.result
        s.on_loop(self._stop, 50)
        aio.run(csv.scheduler().start())
        assert csv.result is not None
        self.assertEqual(len(csv.result), 30_000)

    def test_scatterplot2(self) -> None:
        s = self.scheduler(clean=True)
        with s:
            random = RandomPTable(2, rows=2000_000, throttle=1000, scheduler=s)
            sp = MCScatterPlot(
                scheduler=s, classes=[("Scatterplot", "_1", "_2")], approximate=True
            )
            sp.create_dependent_modules(random, "result", with_sampling=False)
            cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
            cnt.input[0] = random.output.result
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = sp.output.result

        async def fake_input_1(scheduler: Scheduler, rn: int) -> None:
            module = scheduler["variable_1"]
            print("from input variable_1")
            await module.from_input({"x": LOWER_X, "y": LOWER_Y})

        async def fake_input_2(scheduler: Scheduler, rn: int) -> None:
            module = scheduler["variable_2"]
            print("from input variable_2")
            await module.from_input({"x": UPPER_X, "y": UPPER_Y})
        s.on_loop(self._stop, 100)
        s.on_loop(fake_input_1, 30)
        s.on_loop(fake_input_2, 30)
        aio.run(s.start())
        js = sp.to_json()
        x, y, _ = zip(*js["sample"]["data"])
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        self.assertGreaterEqual(min_x, LOWER_X)
        self.assertGreaterEqual(min_y, LOWER_Y)
        self.assertLessEqual(max_x, UPPER_X)
        self.assertLessEqual(max_y, UPPER_Y)


if __name__ == "__main__":
    ProgressiveTest.main()
