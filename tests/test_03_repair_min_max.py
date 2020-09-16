from . import ProgressiveTest
from progressivis import Print, Scheduler
from progressivis.table.module import TableModule
from progressivis.table.table import Table
from progressivis.core.slot import SlotDescriptor
from progressivis.stats import  RandomTable, ScalarMax, Min
from progressivis.table.stirrer import Stirrer
from progressivis.core.bitmap import bitmap
from progressivis.core import aio
from progressivis.core.utils import indices_len, fix_loc
import numpy as np

ScalarMax._reset_calls_counter = 0
ScalarMax._orig_reset = ScalarMax.reset
def _reset_func(self_):
    ScalarMax._reset_calls_counter += 1
    print("RESET")
    return ScalarMax._orig_reset(self_)
ScalarMax.reset = _reset_func

Min._reset_calls_counter = 0
Min._orig_reset = Min.reset
def _reset_func(self_):
    Min._reset_calls_counter += 1
    return Min._orig_reset(self_)
Min.reset = _reset_func


class MyStirrer(TableModule):
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, watched, proc_sensitive=True, mode='delete', **kwds):
        super().__init__(**kwds)
        self.watched = watched
        self.proc_sensitive = proc_sensitive
        self.mode = mode
        self.default_step_size = 100
        self.done = False

    def run_step(self, run_number, step_size, howlong):
        input_slot = self.get_input_slot('table')
        # input_slot.update(run_number)
        steps = 0
        if not input_slot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)
        created = input_slot.created.next(step_size)
        steps = indices_len(created)
        input_table = input_slot.data()
        if self._table is None:
            self._table = Table(self.generate_table_name('stirrer'),
                                dshape=input_table.dshape, )
        v = input_table.loc[fix_loc(created), :]
        self._table.append(v)
        if not self.done:
            #if self.mode != 'delete':
            #    import pdb;pdb.set_trace()
            sensitive_ids = bitmap(self.scheduler().modules()[self.watched]._sensitive_ids.values())
            if sensitive_ids:
                if self.proc_sensitive:
                    if self.mode == 'delete':
                        print('delete sensitive', sensitive_ids)
                        del self._table.loc[sensitive_ids]
                    else:
                        print('update sensitive', sensitive_ids)
                        self._table.loc[sensitive_ids, 0] = 99999.0
                    self.done = True
                else: # non sensitive
                    if len(self._table) > 10:
                        #import pdb;pdb.set_trace()
                        for i in range(10):
                            id_ = self._table.index[i]
                            if id_ not in sensitive_ids:
                                if self.mode == 'delete':
                                    del self._table.loc[id_]
                                else:
                                    self._table.loc[id_, 0] = 99999.0
                                self.done = True

        return self._return_run_step(self.next_state(input_slot),
                                     steps_run=steps)

class TestRepairMax(ProgressiveTest):
    def test_repair_max(self):
        """
        no disruption
        """
        s=Scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        max_=ScalarMax(name='max_'+str(hash(random)), scheduler=s)
        max_.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.table
        aio.run(s.start())
        res1 = random.table().max()
        res2 = max_.table()
        self.compare(res1, res2)

    def test_repair_max2(self):
        """
        runs with sensitive ids deletion
        """
        s=Scheduler()
        ScalarMax._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        max_=ScalarMax(name='max_repair_test2', scheduler=s)
        stirrer = MyStirrer('max_repair_test2', scheduler=s)
        stirrer.input.table = random.output.table
        max_.input.table = stirrer.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.table
        aio.run(s.start())
        self.assertEqual(ScalarMax._reset_calls_counter, 1)
        res1 = stirrer.table().max()
        res2 = max_.table()
        self.compare(res1, res2)
        
    def te_st_repair_max3(self):
        """
        runs with NON-sensitive ids deletion
        """
        s=Scheduler()
        ScalarMax._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        max_=ScalarMax(name='max_repair_test2', scheduler=s)
        stirrer = MyStirrer('max_repair_test2', proc_sensitive=False, scheduler=s)
        stirrer.input.table = random.output.table
        max_.input.table = stirrer.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.table
        aio.run(s.start())
        self.assertEqual(ScalarMax._reset_calls_counter, 0)
        res1 = stirrer.table().max()
        res2 = max_.table()
        self.compare(res1, res2)

    def test_repair_max4(self):
        """
        runs with sensitive ids update
        """
        s=Scheduler()
        ScalarMax._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        max_=ScalarMax(name='max_repair_test2', scheduler=s)
        stirrer = MyStirrer('max_repair_test2', mode='update', scheduler=s)
        stirrer.input.table = random.output.table
        max_.input.table = stirrer.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.table
        aio.run(s.start())
        self.assertEqual(ScalarMax._reset_calls_counter, 1)
        res1 = stirrer.table().max()
        res2 = max_.table()
        self.compare(res1, res2)

    def test_repair_max5(self):
        """
        runs with NON-sensitive ids updates
        """
        s=Scheduler()
        ScalarMax._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        max_=ScalarMax(name='max_repair_test2', scheduler=s)
        stirrer = MyStirrer('max_repair_test2', proc_sensitive=False,
                            mode='update', scheduler=s)
        stirrer.input.table = random.output.table
        max_.input.table = stirrer.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = max_.output.table
        aio.run(s.start())
        self.assertEqual(ScalarMax._reset_calls_counter, 0)
        res1 = stirrer.table().max()
        res2 = max_.table()
        self.compare(res1, res2)

    def compare(self, res1, res2):
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        #print('v1 = ', v1, res1.keys())
        #print('v2 = ', v2, res2.keys())
        self.assertTrue(np.allclose(v1, v2))

if __name__ == '__main__':
    unittest.main()
