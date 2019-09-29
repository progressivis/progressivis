from . import ProgressiveTest, skip

from progressivis import Print
from progressivis.io.input import Input
from progressivis.core.utils import Thread
import numpy as np
from time import sleep


def _ten_times(scheduler, run_number):
    print('ten_times %d' % run_number)
    if run_number > 20:
        scheduler.stop()


def _do_line(inp, s):
    sleep(2)
    for r in range(10):
        inp.from_input('line#%d' % r)
        sleep(np.random.random())
    sleep(1)
    s.stop()


class TestInput(ProgressiveTest):
    def test_input(self):
        s = self.scheduler()
        with s:
            inp = Input(scheduler=s)
            pr = Print(proc=self.terse, scheduler=s)
            pr.input.df = inp.output.table
        t = Thread(target=_do_line, args=(inp, s))
        t.start()
        s.start()
        s.join()
        self.assertEqual(len(inp.table()), 10)


if __name__ == '__main__':
    ProgressiveTest.main()
