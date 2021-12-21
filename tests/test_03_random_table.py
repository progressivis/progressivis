from . import ProgressiveTest
from progressivis.core import aio
from progressivis import Every
from progressivis.stats.random_table import RandomTable


def print_len(x):
    if x is not None:
        print(len(x))


class TestRandomTable(ProgressiveTest):
    def test_random_table(self):
        s = self.scheduler()
        module = RandomTable(["a", "b"], rows=10000, scheduler=s)
        self.assertEqual(module.result.columns[0], "a")
        self.assertEqual(module.result.columns[1], "b")
        self.assertEqual(len(module.result.columns), 2)  # add the UPDATE_COLUMN
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = module.output.result
        aio.run(s.start())
        # s.join()
        self.assertEqual(len(module.result), 10000)

    #        self.assertFalse(module.result['a'].isnull().any())
    #        self.assertFalse(module.result['b'].isnull().any())

    def test_random_table2(self):
        s = self.scheduler()
        # produces more than 4M rows per second on my laptop
        module = RandomTable(10, rows=1000000, force_valid_ids=True, scheduler=s)
        self.assertEqual(len(module.result.columns), 10)  # add the UPDATE_COLUMN
        self.assertEqual(module.result.columns[0], "_1")
        self.assertEqual(module.result.columns[1], "_2")
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = module.output.result
        aio.run(s.start())
        # s.join()
        self.assertEqual(len(module.result), 1000000)


#        self.assertFalse(module.result['_1'].isnull().any())
#        self.assertFalse(module.result['_2'].isnull().any())
