from . import ProgressiveTest
import numpy as np
from progressivis import Print
from progressivis.stats import IdxMax, IdxMin, Max, Min, RandomTable
from progressivis.table.stirrer import Stirrer
from progressivis.core import aio



class TestIdxMax(ProgressiveTest):
    def tearDown(self):
        TestIdxMax.cleanup()
    def test_idxmax(self):
        s=self.scheduler()
        random = RandomTable(10, rows=10000,throttle=1000, scheduler=s)
        idxmax=IdxMax(scheduler=s)
        idxmax.input.table = random.output.result
        max_=Max(scheduler=s)
        max_.input.table = random.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = idxmax.output.result
        aio.run(s.start())
        max1 = max_.result
        #print('max1', max1)
        max2 = idxmax.max().last().to_dict()
        #print('max2', max2)
        self.compare(max1, max2)

    def test_idxmax2(self):
        s=self.scheduler()
        random = RandomTable(10, rows=10000,throttle=1000, scheduler=s)
        stirrer = Stirrer(update_column='_1', delete_rows=5,
                          fixed_step_size=100, scheduler=s)
        stirrer.input.table = random.output.result
        idxmax=IdxMax(scheduler=s)
        idxmax.input.table = stirrer.output.result
        max_=Max(scheduler=s)
        max_.input.table = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = idxmax.output.result
        aio.run(s.start())
        #import pdb;pdb.set_trace()
        max1 = max_.result
        #print('max1', max1)
        max2 = idxmax.max().last().to_dict()
        #print('max2', max2)
        self.compare(max1, max2)

    def test_idxmin(self):
        s=self.scheduler()
        random = RandomTable(10, rows=10000,throttle=1000, scheduler=s)
        idxmin=IdxMin(scheduler=s)
        idxmin.input.table = random.output.result
        min_=Min(scheduler=s)
        min_.input.table = random.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = idxmin.output.result
        aio.run(s.start())
        min1 = min_.result
        #print('min1', min1)
        min2 = idxmin.min().last().to_dict()
        #print('min2', min2)
        self.compare(min1, min2)

    def test_idxmin2(self):
        s=self.scheduler()
        random = RandomTable(10, rows=10000,throttle=1000, scheduler=s)
        stirrer = Stirrer(update_column='_1', delete_rows=5,
                          fixed_step_size=100, scheduler=s)
        stirrer.input.table = random.output.result
        idxmin=IdxMin(scheduler=s)
        idxmin.input.table = stirrer.output.result
        min_=Min(scheduler=s)
        min_.input.table = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = idxmin.output.result
        aio.run(s.start())
        min1 = min_.result
        #print('min1', min1)
        min2 = idxmin.min().last().to_dict()
        #print('min2', min2)
        self.compare(min1, min2)

    def compare(self, res1, res2):
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        #print('v1 = ', v1, res1.keys())
        #print('v2 = ', v2, res2.keys())
        self.assertTrue(np.allclose(v1, v2))
if __name__ == '__main__':
    ProgressiveTest.main()
