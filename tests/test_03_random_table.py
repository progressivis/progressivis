from . import ProgressiveTest

from progressivis import Every
from progressivis.stats.random_table import RandomTable

def print_len(x):
    if x is not None:
        print(len(x))


class TestRandomTable(ProgressiveTest):
    def test_random_table(self):
        s = self.scheduler()
        module=RandomTable(['a', 'b'], rows=10000, scheduler=s)
        self.assertEqual(module.table().columns[0],'a')
        self.assertEqual(module.table().columns[1],'b')
        self.assertEqual(len(module.table().columns), 2) # add the UPDATE_COLUMN
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input.df = module.output.table
        s.start()
        s.join()
        self.assertEqual(len(module.table()), 10000)
#        self.assertFalse(module.table()['a'].isnull().any())
#        self.assertFalse(module.table()['b'].isnull().any())

    def test_random_table2(self):
        s = self.scheduler()
        # produces more than 4M rows per second on my laptop
        module=RandomTable(10, rows=1000000, force_valid_ids=True, scheduler=s)
        self.assertEqual(len(module.table().columns), 10) # add the UPDATE_COLUMN
        self.assertEqual(module.table().columns[0],'_1')
        self.assertEqual(module.table().columns[1],'_2')
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input.df = module.output.table
        s.start()
        s.join()        
        self.assertEqual(len(module.table()), 1000000)
#        self.assertFalse(module.table()['_1'].isnull().any())
#        self.assertFalse(module.table()['_2'].isnull().any())
