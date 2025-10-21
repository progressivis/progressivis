from . import ProgressiveTest

from progressivis import Tick
from progressivis.io.input import Input
import numpy as np
from progressivis.core.api import Scheduler
from progressivis.core import aio


async def _do_line(inp: Input, s: Scheduler) -> None:
    await aio.sleep(2)
    for r in range(10):
        await inp.from_input({"input": f"line#{r}"})
        await aio.sleep(np.random.random())
    await aio.sleep(1)
    await s.stop()


class TestInput(ProgressiveTest):
    def test_input(self) -> None:
        s = self.scheduler
        with s:
            inp = Input(scheduler=s)
            pr = Tick(scheduler=s)
            pr.input.df = inp.output.result
        aio.run_gather(s.start(), _do_line(inp, s))
        assert inp.result is not None
        self.assertEqual(len(inp.result), 10)


if __name__ == "__main__":
    ProgressiveTest.main()
