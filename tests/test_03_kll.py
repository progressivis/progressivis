from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print
from progressivis.stats.kll import KLLSketch
from progressivis.stats import RandomTable
import numpy as np
from datasketches import kll_floats_sketch

K = 200
BINS = 128
QUANTILE = 0.5

class TestKll(ProgressiveTest):
    def test_kll(self):
        s = self.scheduler()
        random = RandomTable(3, rows=10_000, scheduler=s)
        kll=KLLSketch(column='_1', scheduler=s)
        kll.input[0] = random.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = kll.output.result
        aio.run(s.start())
        val = random.result['_1'].value
        sk = kll_floats_sketch(K)
        sk.update(val)
        self.assertAlmostEqual(kll.result['max'], sk.get_max_value())
        self.assertAlmostEqual(kll.result['min'], sk.get_min_value())        
        self.assertAlmostEqual(kll.result['quantile'], sk.get_quantile(QUANTILE))        


if __name__ == '__main__':
    ProgressiveTest.main()
