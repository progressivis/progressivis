import unittest

from progressivis import Print, Scheduler
from progressivis.io import Input

import pandas as pd
import numpy as np
import threading
from time import sleep

def print_len(x):
    if x is not None:
        print len(x)

def ten_times(scheduler, run_number):
    if run_number > 10:
        scheduler.stop()

def do_line(inp,s):
    sleep(2)
    for r in xrange(10):
        inp.add_input('line#%d'%r)
        sleep(np.random.random())
    sleep(3)
    s.stop()

class TestInput(unittest.TestCase):
    def test_input(self):
        s=Scheduler()
        inp = Input(scheduler=s)
        pr=Print(scheduler=s)
        pr.input.inp = inp.output.df
        #s.start(ten_times)
        t=threading.Thread(target=do_line,args=(inp,s))
        t.start()
        s.start()
        self.assertEqual(len(inp.df()), 10)

if __name__ == '__main__':
    unittest.main()
