from . import ProgressiveTest
import asyncio as aio
from progressivis import Print, Scheduler
from progressivis.stats import  RandomTable, Max
from progressivis.table.dummymod import DummyMod
from progressivis.core.bitmap import bitmap

import numpy as np

class TestDummy(ProgressiveTest):
    def test_dummy(self):
        s=Scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        dummy_ = DummyMod(update_column='_1', delete_rows=5, update_rows=5, fixed_step_size=100, scheduler=s)
        dummy_.input.table = random.output.table
        max_=Max(name='max_'+str(hash(random)), scheduler=s)
        max_.input.table = dummy_.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.table
        aio.run(s.start())
        #idx = dummy_.get_input_slot('table').data().eval('_1>0.5', result_object='index')
        #self.assertEqual(dummy_._table.selection, bitmap(idx))

if __name__ == '__main__':
    unittest.main()
