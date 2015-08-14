import unittest

from progressivis import *
from progressivis.io.vec_loader import *

import os
import numpy as np
from pprint import pprint

    
class TestProgressiveLoadVEC(unittest.TestCase):
    filename='data/warlogs.vec.bz2'

    def test_read_vec(self):
        module=VECLoader(self.filename,id='test_read_vec')
        self.assertTrue(module.df() is None)
        module.run(0)
        s = module.trace_stats(max_runs=1)
        df = module.df()
        self.assertFalse(df is None)
        l = len(df)
        self.assertEqual(l, len(df[module.update_timestamps()==module.last_update()]))
        cnt = 1
        
        while not module.is_terminated():
            module.run(cnt)
            cnt += 1
            s = module.trace_stats(max_runs=1)
            df = module.df()
            ln = len(df)
            print "Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), ln)
            self.assertEqual(ln-l, len(df[module.update_timestamps()==module.last_update()]))
            l =  ln
        s = module.trace_stats(max_runs=1)
        print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        df2 = module.df().groupby([Module.UPDATE_COLUMN])
        self.assertEqual(cnt, len(df2))

suite = unittest.TestLoader().loadTestsFromTestCase(TestProgressiveLoadVEC)

if __name__ == '__main__':
    unittest.main()
