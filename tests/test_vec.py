import unittest

from progressivis import *
from progressivis.io.vec_loader import *
from progressivis.datasets import get_dataset

    
class TestProgressiveLoadVEC(unittest.TestCase):
    def setUp(self):
        self.scheduler = Scheduler()

    def test_read_vec(self):
        module=VECLoader(get_dataset('warlogs'),
                         id='test_read_vec')
        self.assertTrue(module.df() is None)
        module.run(0)
        s = module.trace_stats(max_runs=1)
        df = module.df()
        self.assertFalse(df is None)
        l = len(df)
        self.assertEqual(l, len(df[df[module.UPDATE_COLUMN]==module.last_update()]))
        cnt = 1
        
        while not module.is_zombie():
            module.run(cnt)
            cnt += 1
            s = module.trace_stats(max_runs=1)
            df = module.df()
            ln = len(df)
            print "Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), ln)
            self.assertEqual(ln-l, len(df[df[module.UPDATE_COLUMN]==module.last_update()]))
            l =  ln
        s = module.trace_stats(max_runs=1)
        print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        df2 = module.df().groupby([Module.UPDATE_COLUMN])
        self.assertEqual(cnt, len(df2))

if __name__ == '__main__':
    unittest.main()
