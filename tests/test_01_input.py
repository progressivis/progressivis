from . import ProgressiveTest

from progressivis import Print
from progressivis.io.input import Input
import numpy as np
from progressivis.core import aio

async def _do_line(inp, s):
    await aio.sleep(2)
    for r in range(10):
        inp.from_input('line#%d' % r)
        await aio.sleep(np.random.random())
    await aio.sleep(1)
    await s.stop()


class TestInput(ProgressiveTest):
    def test_input(self):
        s = self.scheduler()
        with s:
            inp = Input(scheduler=s)
            pr = Print(proc=self.terse, scheduler=s)
            pr.input.df = inp.output.result
        aio.run_gather(s.start(), _do_line(inp, s))
        self.assertEqual(len(inp.result), 10)


if __name__ == '__main__':
    ProgressiveTest.main()
