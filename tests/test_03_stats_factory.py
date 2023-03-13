from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print, Scheduler
from progressivis.vis import StatsFactory, DataShape
from progressivis.stats import RandomPTable
import numpy as np
import pandas as pd
from typing import Any, Dict
from progressivis.stats.blobs_table import BlobsPTable
from progressivis.core import Sink


async def fake_input(sched: Scheduler, name: str, t: float, inp: Dict[str, Any]) -> Any:
    await aio.sleep(t)
    module = sched.modules()[name]
    await module.from_input(inp)


def my_stop(s: Scheduler, _: Any) -> None:
    s.task_stop()


funcs = ["hide", "hist", "min", "max", "var"]
arr = np.zeros((3, len(funcs)), dtype=object)
matrix_max = pd.DataFrame(arr.copy(), index=["_1", "_2", "_3"], columns=funcs)
matrix_max.loc["_2", "max"] = True

matrix_hist = pd.DataFrame(arr.copy(), index=["A", "B", "C"], columns=funcs)
matrix_hist.loc["B", "hist"] = True


class TestStatsFactory(ProgressiveTest):
    def test_datashape(self) -> None:
        np.random.seed(42)
        s = self.scheduler()
        random = RandomPTable(3, rows=10_000, scheduler=s)
        ds = DataShape(scheduler=s)
        ds.input.table = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = ds.output.result
        aio.run(s.start())
        print(s.modules())

    def test_sf(self) -> None:
        np.random.seed(42)
        s = self.scheduler()
        random = RandomPTable(3, rows=10_000, scheduler=s)
        sf = StatsFactory(input_module=random, scheduler=s)
        sf.create_dependent_modules(var_name="my_dyn_var")
        sf.input.table = random.output.result
        sink = Sink(scheduler=s)
        # sink.input.inp = random.output.result
        sink.input.inp = sf.output.result

        async def fake_input_1(scheduler: Scheduler, rn: int) -> None:
            module = scheduler["my_dyn_var"]
            print("from input my_dyn_var", "test_sf")
            await module.from_input({"matrix": matrix_max})

        s.on_loop(my_stop, 4)
        s.on_loop(fake_input_1, 3)
        aio.run(s.start())
        print(s.modules())

    def test_pattern(self) -> None:
        s = self.scheduler()
        n_samples = 1_000
        centers = [(0.1, 0.3, 0.5), (0.7, 0.5, 3.3), (-0.4, -0.3, -11.1)]
        cols = ["A", "B", "C"]
        with s:
            data = BlobsPTable(
                columns=cols,
                centers=centers,
                cluster_std=0.2,
                rows=n_samples,
                scheduler=s,
            )
            # ds = DataShape(scheduler=s)
            # ds.input.table = data.output.result
            factory = StatsFactory(input_module=data, scheduler=s)
            factory.create_dependent_modules(var_name="my_dyn_var")
            factory.input.table = data.output.result
            sink = Sink(scheduler=s)
            # sink.input.inp = ds.output.result
            sink.input.inp = factory.output.result

        async def fake_input_1(scheduler: Scheduler, rn: int) -> None:
            module = scheduler["my_dyn_var"]
            print("from input my_dyn_var")
            await module.from_input({"matrix": matrix_hist})

        s.on_loop(my_stop, 4)
        s.on_loop(fake_input_1, 3)
        aio.run(s.start())


if __name__ == "__main__":
    ProgressiveTest.main()
