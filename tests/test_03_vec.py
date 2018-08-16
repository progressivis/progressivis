from . import ProgressiveTest, skip

from progressivis.io import VECLoader
from progressivis.datasets import get_dataset

    
class TestProgressiveLoadVEC(ProgressiveTest):

    @skip("Need to implement sparse columns")
    def test_read_vec(self):
        module=VECLoader(get_dataset('warlogs'),
                         name='test_read_vec')
        #self.assertTrue(module.table() is None)
        module.run(0)
        s = module.trace_stats(max_runs=1)
        df = module.table()
        self.assertFalse(df is None)
        l = len(df)
        #self.assertEqual(l, len(df[df[UPDATE_COLUMN]==module.last_update()]))
        cnt = 1
        
        while not module.is_zombie():
            module.run(cnt)
            cnt += 1
            s = module.trace_stats(max_runs=1)
            df = module.table()
            ln = len(df)
            #print ("Run time: %gs, loaded %d rows" % (s['duration'][len(s)-1], ln))
            #self.assertEqual(ln-l, len(df[df[UPDATE_COLUMN]==module.last_update()]))
            l =  ln
        s = module.trace_stats(max_runs=1)
        ln = len(module.table())
        #print("Done. Run time: %gs, loaded %d rows" % (s['duration'][len(s)-1], ln))
        #df2 = module.df().groupby([UPDATE_COLUMN])
        #self.assertEqual(cnt, len(df2))

if __name__ == '__main__':
    ProgressiveTest.main()
