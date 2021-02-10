from . import ProgressiveTest, skip, skipIf

from progressivis import Print
from progressivis.stats  import RandomTable
from progressivis.stats.cxxmax import Max, CxxMax

import numpy as np

class TestCxxMax(ProgressiveTest):

    def compare(self, res1, res2):
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        #print('v1 = ', v1)
        #print('v2 = ', v2)
        self.assertTrue(np.allclose(v1, v2))

    @skipIf(CxxMax is None, "C++ module is missing")
    def test_max(self):
        s = self.scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        max_=Max(name='max_'+str(hash(random)), scheduler=s)
        max_.input[0] = random.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.result
        s.start()
        s.join()
        res1 = random.table().max()
        res2 = max_.cxx_module.get_output_table().last()
        self.compare(res1, res2)


if __name__ == '__main__':
    ProgressiveTest.main()
