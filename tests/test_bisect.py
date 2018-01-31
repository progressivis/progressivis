from . import ProgressiveTest

from progressivis import Print, Scheduler
from progressivis.stats import  RandomTable
from progressivis.table.bisectmod import BisectMod
from progressivis.core.bitmap import bitmap

import numpy as np

class TestBisect(ProgressiveTest):
    def test_bisect(self):
        s=Scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        bisect_ = BisectMod(column='_1', pivot=0.5, op='>', cache=(0.0, 1.0, 10), scheduler=s)
        bisect_.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = bisect_.output.table
        s.start()
        s.join()
        idx = bisect_.get_input_slot('table').data().eval('_1>0.5', result_object='index')
        self.assertEqual(bisect_._table.selection, bitmap(idx))


if __name__ == '__main__':
    unittest.main()
