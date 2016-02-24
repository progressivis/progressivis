import unittest

from progressivis import Print, Scheduler, log_level
from progressivis.io import Input

import pandas as pd
import numpy as np
import threading
from time import sleep

#log_level()

def ten_times(scheduler, run_number):
    print 'ten_times %d'%run_number
    if run_number > 20:
        scheduler.stop()

def do_line(inp,s):
    sleep(2)
    for r in xrange(10):
        inp.from_input('line#%d'%r)
        sleep(np.random.random())
    sleep(1)
    s.stop()

class TestInput(unittest.TestCase):
    def test_input(self):
        s=Scheduler()
        inp = Input(scheduler=s)
        pr=Print(scheduler=s)
        pr.input.df = inp.output.df
        t=threading.Thread(target=do_line,args=(inp,s))
        t.start()
        s.start()
        self.assertEqual(len(inp.df()), 10)

if __name__ == '__main__':
    unittest.main()
