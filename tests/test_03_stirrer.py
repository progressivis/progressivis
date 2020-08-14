from . import ProgressiveTest
from progressivis import Print, Scheduler
from progressivis.stats import  RandomTable, Max, Min
from progressivis.table.stirrer import Stirrer
from progressivis.core.bitmap import bitmap
from progressivis.core import aio

import numpy as np

Max._reset_calls_counter = 0
Max._orig_reset = Max.reset
def _reset_func(self_):
    Max._reset_calls_counter += 1
    return Max._orig_reset(self_)
Max.reset = _reset_func

Min._reset_calls_counter = 0
Min._orig_reset = Min.reset
def _reset_func(self_):
    Min._reset_calls_counter += 1
    return Min._orig_reset(self_)
Min.reset = _reset_func

class TestStirrer(ProgressiveTest):
    def test_stirrer(self):
        s=Scheduler()
        Max._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(update_column='_1',
                          delete_rows=5, update_rows=5,
                          fixed_step_size=100, scheduler=s)
        stirrer.input.table = random.output.table
        max_=Max(name='max_'+str(hash(random)), scheduler=s)
        max_.input.table = stirrer.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.table
        aio.run(s.start())
        self.assertNotEqual(Max._reset_calls_counter, 0)
        res1 = random.table().max()
        res2 = max_.table()
        self.compare(res1, res2)

    def test_stirrer2(self):
        s=Scheduler()
        Max._reset_calls_counter = 0
        random = RandomTable(5, rows=1000000, scheduler=s)
        stirrer = Stirrer(update_column='_1',
                          update_rows=5,
                          fixed_step_size=100, scheduler=s)
        stirrer.input.table = random.output.table
        max_=Max(name='max_'+str(hash(random)),
                 columns=['_2', '_3', '_4'], scheduler=s)
        max_.input.table = stirrer.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.table
        aio.run(s.start())
        self.assertEqual(Max._reset_calls_counter, 0)
        res1 = random.table().loc[:,1:4].max()
        res2 = max_.table()
        self.compare(res1, res2)

    def test_stirrer3(self):
        s=Scheduler()
        Min._reset_calls_counter = 0
        random = RandomTable(5, rows=1000000, scheduler=s)
        stirrer = Stirrer(update_column='_1',
                          update_rows=5,
                          fixed_step_size=100, scheduler=s)
        stirrer.input.table = random.output.table
        min_=Min(name='min_'+str(hash(random)),
                 columns=['_2', '_3', '_4'], scheduler=s)
        min_.input.table = stirrer.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = min_.output.table
        aio.run(s.start())
        self.assertEqual(Min._reset_calls_counter, 0)
        res1 = random.table().loc[:,1:4].min()
        res2 = min_.table()
        self.compare(res1, res2)

    def compare(self, res1, res2):
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        #print('v1 = ', v1, res1.keys())
        #print('v2 = ', v2, res2.keys())
        self.assertTrue(np.allclose(v1, v2))

if __name__ == '__main__':
    unittest.main()
