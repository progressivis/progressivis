from . import ProgressiveTest

import asyncio
from progressivis import Print
from progressivis.stats import Min, Max, RandomTable

import numpy as np

class TestMinMax(ProgressiveTest):
    def test_min(self):
        s = self.scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        min_=Min(name='min_'+str(hash(random)), scheduler=s)
        min_.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = min_.output.table
        asyncio.run(s.start())
        #s.join()
        res1 = random.table().min()
        res2 = min_.table().last()
        self.compare(res1, res2)

    def compare(self, res1, res2):
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        #print('v1 = ', v1)
        #print('v2 = ', v2)
        self.assertTrue(np.allclose(v1, v2))

    def test_max(self):
        s = self.scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        max_=Max(name='max_'+str(hash(random)), scheduler=s)
        max_.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.table
        asyncio.run(s.start())
        #s.join()
        res1 = random.table().max()
        res2 = max_.table().last()
        self.compare(res1, res2)


if __name__ == '__main__':
    ProgressiveTest.main()
