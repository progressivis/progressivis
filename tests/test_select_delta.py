import unittest

from progressivis import Scheduler, Every, log_level
from progressivis.io import AddToRow
from progressivis.core.select_delta import SelectDelta

import pandas as pd
import numpy as np

class TestSelectDelta(unittest.TestCase):
    def test_select_delta(self):
        #log_level()
        delta = np.array([0, 0.05])
        points = [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
        s=Scheduler()
        df=pd.DataFrame(points)
        add_to_row=AddToRow(df, scheduler=s)
        def tick_proc(s, run_number):
            if run_number > 100:
                s.stop()
            #print add_to_row.df()
            try:
                add_to_row.from_input({1: delta})
            except Exception as e:
                print 'Error: %s'%e
        q=SelectDelta(delta=0.5,scheduler=s)
        q.input.df = add_to_row.output.df
        prlen = Every(scheduler=s)
        prlen.input.df = q.output.df
        s.start(tick_proc=tick_proc)
        self.assertEqual(len(q.df()), 3)



if __name__ == '__main__':
    #import cProfile
    #cProfile.run('unittest.main()', sort='cumulative')
    unittest.main()
