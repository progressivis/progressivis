from __future__ import annotations

from progressivis import Every, Print, Scheduler
from progressivis.io import CSVLoader
from progressivis.vis import MCScatterPlot
from progressivis.datasets import get_dataset
from progressivis.stats import RandomTable
from progressivis.core import aio
from . import ProgressiveTest

from typing import Any, Dict


def print_len(x: Any) -> None:
    if x is not None:
        print(len(x))


def print_repr(x: Any) -> None:
    if x is not None:
        print(repr(x))


# async def idle_proc(s, _):
#    await s.stop()


LOWER_X = 0.2
LOWER_Y = 0.3
UPPER_X = 0.8
UPPER_Y = 0.7


async def fake_input(sched: Scheduler, name: str, t: float, inp: Dict[str, Any]) -> Any:
    await aio.sleep(t)
    module = sched.modules()[name]
    await module.from_input(inp)


async def sleep_then_stop(s: Scheduler, t: float) -> None:
    await aio.sleep(t)
    await s.stop()
    print(s._run_list)


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
            # sts = sleep_then_stop(s, 5)
        s.on_loop(self._stop, 5)
        aio.run(csv.scheduler().start())
        self.assertEqual(len(csv.table), 30000)

    def test_scatterplot2(self) -> None:
        s = self.scheduler(clean=True)
        with s:
            random = RandomTable(2, rows=2000000, throttle=1000, scheduler=s)
            sp = MCScatterPlot(
                scheduler=s, classes=[("Scatterplot", "_1", "_2")], approximate=True
            )
            sp.create_dependent_modules(random, "result", with_sampling=False)
            cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
            cnt.input[0] = random.output.result
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = sp.output.result

        async def fake_input_1(scheduler: Scheduler, rn: int) -> None:
            module = scheduler["dyn_var_1"]
            print("from input dyn_var_1")
            await module.from_input({"x": LOWER_X, "y": LOWER_Y})

        async def fake_input_2(scheduler: Scheduler, rn: int) -> None:
            module = scheduler["dyn_var_2"]
            print("from input dyn_var_2")
            await module.from_input({"x": UPPER_X, "y": UPPER_Y})

        # finp1 = fake_input(s, "dyn_var_1", 6, {"x": LOWER_X, "y": LOWER_Y})
        # finp2 = fake_input(s, "dyn_var_2", 6, {"x": UPPER_X, "y": UPPER_Y})
        # sts = sleep_then_stop(s, 10)
        s.on_loop(self._stop, 10)
        # s.on_loop(prt)
        s.on_loop(fake_input_1, 3)
        s.on_loop(fake_input_2, 3)
        # aio.run_gather(sp.scheduler().start(), sts)
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
