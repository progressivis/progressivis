from . import ProgressiveTest
from progressivis import Print, Scheduler
from progressivis.stats import  RandomTable, Max
from progressivis.table.stirrer import Stirrer
from progressivis.core.bitmap import bitmap
from progressivis.core import aio

import numpy as np

class TestStirrer(ProgressiveTest):
    def test_stirrer(self):
        s=Scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(update_column='_1', delete_rows=5, update_rows=5, fixed_step_size=100, scheduler=s)
        stirrer.input.table = random.output.result
        max_=Max(name='max_'+str(hash(random)), scheduler=s)
        max_.input.table = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.result
        aio.run(s.start())
        #idx = dummy_.get_input_slot('table').data().eval('_1>0.5', result_object='index')
        #self.assertEqual(dummy_._table.selection, bitmap(idx))

if __name__ == '__main__':
    unittest.main()
