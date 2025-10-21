from __future__ import annotations

from . import ProgressiveTest

from progressivis import Tick, VECLoader, CSVLoader

# from progressivis.metrics import PairwiseDistances
from progressivis.datasets import get_dataset
from progressivis.core import aio

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis import Scheduler


def print_len(x: Any) -> None:
    if x is not None:
        print(len(x))


times = 0


async def ten_times(scheduler: Scheduler, run_number: int) -> None:
    global times
    times += 1
    if times > 10:
        await scheduler.stop()


class TestPairwiseDistances(ProgressiveTest):
    def NOtest_vec_distances(self) -> None:
        s = self.scheduler
        vec = VECLoader(get_dataset("warlogs"), scheduler=s)
        cnt = Tick(scheduler=s)
        cnt.input[0] = vec.output.result
        global times
        times = 0
        aio.run(s.start())
        _ = vec.result

    def test_csv_distances(self) -> None:
        s = self.scheduler
        vec = CSVLoader(
            get_dataset("smallfile"), header=None, scheduler=s
        )
        cnt = Tick(oscheduler=s)
        cnt.input[0] = vec.output.result
        global times
        times = 0
        aio.run(s.start(ten_times))
        _ = vec.result


if __name__ == "__main__":
    ProgressiveTest.main()
