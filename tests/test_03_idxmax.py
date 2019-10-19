from . import ProgressiveTest
import numpy as np
from progressivis import Print
from progressivis.stats import IdxMax, IdxMin, Max, Min, RandomTable
import asyncio as aio

class TestIdxMax(ProgressiveTest):
    def tearDown(self):
        TestIdxMax.cleanup()
    def test_idxmax(self):
        s=self.scheduler()
        random = RandomTable(10, rows=10000,throttle=1000, scheduler=s)
        idxmax=IdxMax(scheduler=s)
        idxmax.input.table = random.output.table
        max_=Max(scheduler=s)
        max_.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = idxmax.output.max
        aio.run(s.start())
        max1 = max_.table().last().to_dict()
        #print('max1', max1)
        max2 = idxmax.max().last().to_dict()
        #print('max2', max2)
        self.assertAlmostEqual(max1, max2)

    def test_idxmin(self):
        s=self.scheduler()
        random = RandomTable(10, rows=10000,throttle=1000, scheduler=s)
        idxmin=IdxMin(scheduler=s)
        idxmin.input.table = random.output.table
        min_=Min(scheduler=s)
        min_.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = idxmin.output.min
        aio.run(s.start())
        min1 = min_.table().last().to_dict()
        #print('min1', min1)
        min2 = idxmin.min().last().to_dict()
        #print('min2', min2)
        self.assertAlmostEqual(min1, min2)

if __name__ == '__main__':
    ProgressiveTest.main()
