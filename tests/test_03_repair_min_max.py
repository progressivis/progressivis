from . import ProgressiveTest, skip, skipIf
from progressivis import Print, Scheduler
from progressivis.table.module import TableModule
from progressivis.table.table import Table
from progressivis.core.slot import SlotDescriptor
from progressivis.stats import  RandomTable, ScalarMax, ScalarMin
from progressivis.table.stirrer import Stirrer
from progressivis.core.bitmap import bitmap
from progressivis.core import aio
from progressivis.core.utils import indices_len, fix_loc
import numpy as np

ScalarMax._reset_calls_counter = 0
ScalarMax._orig_reset = ScalarMax.reset
def _reset_func(self_):
    ScalarMax._reset_calls_counter += 1
    return ScalarMax._orig_reset(self_)
ScalarMax.reset = _reset_func

ScalarMin._reset_calls_counter = 0
ScalarMin._orig_reset = ScalarMin.reset
def _reset_func(self_):
    ScalarMin._reset_calls_counter += 1
    return ScalarMin._orig_reset(self_)
ScalarMin.reset = _reset_func


class MyStirrer(TableModule):
    inputs = [SlotDescriptor('table', type=Table, required=True)]

    def __init__(self, watched, proc_sensitive=True, mode='delete', value=9999.0, **kwds):
        super().__init__(**kwds)
        self.watched = watched
        self.proc_sensitive = proc_sensitive
        self.mode = mode
        self.default_step_size = 100
        self.value = value
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
        if self.result is None:
            self.result = Table(self.generate_table_name('stirrer'),
                                dshape=input_table.dshape, )
        v = input_table.loc[fix_loc(created), :]
        self.result.append(v)
        if not self.done:
            sensitive_ids = bitmap(self.scheduler().modules()[self.watched]._sensitive_ids.values())
            if sensitive_ids:
                if self.proc_sensitive:
                    if self.mode == 'delete':
                        #print('delete sensitive', sensitive_ids)
                        del self.result.loc[sensitive_ids]
                    else:
                        #print('update sensitive', sensitive_ids)
                        self.result.loc[sensitive_ids, 0] = self.value
                    self.done = True
                else: # non sensitive
                    if len(self.result) > 10:
                        for i in range(10):
                            id_ = self.result.index[i]
                            if id_ not in sensitive_ids:
                                if self.mode == 'delete':
                                    del self.result.loc[id_]
                                else:
                                    self.result.loc[id_, 0] = self.value
                                self.done = True

        return self._return_run_step(self.next_state(input_slot),
                                     steps_run=steps)
#@skip
class TestRepairMax(ProgressiveTest):
    def test_repair_max(self):
        """
        test_repair_max()
        max without deletes/updates
        """
        s=Scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        max_=ScalarMax(name='max_'+str(hash(random)), scheduler=s)
        max_.input[0] = random.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        res1 = random.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_repair_max2(self):
        """
        test_repair_max2()
        runs with sensitive ids deletion
        """
        s=Scheduler()
        ScalarMax._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        max_=ScalarMax(name='max_repair_test2', scheduler=s)
        stirrer = MyStirrer(watched='max_repair_test2', scheduler=s)
        stirrer.input[0] = random.output.result
        max_.input[0] = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        self.assertEqual(ScalarMax._reset_calls_counter, 1)
        res1 = stirrer.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_repair_max3(self):
        """
        test_repair_max3()
        runs with NON-sensitive ids deletion
        """
        s=Scheduler()
        ScalarMax._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        max_=ScalarMax(name='max_repair_test3', scheduler=s)
        stirrer = MyStirrer(watched='max_repair_test3', proc_sensitive=False, scheduler=s)
        stirrer.input[0] = random.output.result
        max_.input[0] = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        self.assertEqual(ScalarMax._reset_calls_counter, 0)
        res1 = stirrer.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_repair_max4(self):
        """
        test_repair_max4()
        runs with sensitive ids update
        """
        s=Scheduler()
        ScalarMax._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        max_=ScalarMax(name='max_repair_test4', scheduler=s)
        stirrer = MyStirrer(watched='max_repair_test4', mode='update', value=9999.0, scheduler=s)
        stirrer.input[0] = random.output.result
        max_.input[0] = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        self.assertEqual(ScalarMax._reset_calls_counter, 0)
        res1 = stirrer.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_repair_max5(self):
        """
        test_repair_max5()
        runs with sensitive ids update (critical)
        """
        s=Scheduler()
        ScalarMax._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        max_=ScalarMax(name='max_repair_test4', scheduler=s)
        stirrer = MyStirrer(watched='max_repair_test4', mode='update', value=-9999.0, scheduler=s)
        stirrer.input[0] = random.output.result
        max_.input[0] = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        self.assertEqual(ScalarMax._reset_calls_counter, 1)
        res1 = stirrer.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_repair_max6(self):
        """
        test_repair_max6()
        runs with NON-sensitive ids updates
        """
        s=Scheduler()
        ScalarMax._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        max_=ScalarMax(name='max_repair_test5', scheduler=s)
        stirrer = MyStirrer(watched='max_repair_test5', proc_sensitive=False,
                            mode='update', scheduler=s)
        stirrer.input[0] = random.output.result
        max_.input[0] = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        self.assertEqual(ScalarMax._reset_calls_counter, 0)
        res1 = stirrer.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def compare(self, res1, res2):
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        #print('v1 = ', v1, res1.keys())
        #print('v2 = ', v2, res2.keys())
        self.assertTrue(np.allclose(v1, v2))


class TestRepairMin(ProgressiveTest):
    def test_repair_min(self):
        """
        test_repair_min()
        min without deletes/updates
        """
        s=Scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        min_=ScalarMin(name='min_'+str(hash(random)), scheduler=s)
        min_.input[0] = random.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        res1 = random.result.min()
        res2 = min_.result
        self.compare(res1, res2)
    def test_repair_min2(self):
        """
        test_repair_min2()
        runs with sensitive ids deletion
        """
        s=Scheduler()
        ScalarMin._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        min_=ScalarMin(name='min_repair_test2', scheduler=s)
        stirrer = MyStirrer(watched='min_repair_test2', scheduler=s)
        stirrer.input[0] = random.output.result
        min_.input[0] = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        self.assertEqual(ScalarMin._reset_calls_counter, 1)
        res1 = stirrer.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def test_repair_min3(self):
        """
        test_repair_min3()
        runs with NON-sensitive ids deletion
        """
        s=Scheduler()
        ScalarMin._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        min_=ScalarMin(name='min_repair_test3', scheduler=s)
        stirrer = MyStirrer(watched='min_repair_test3', proc_sensitive=False, scheduler=s)
        stirrer.input[0] = random.output.result
        min_.input[0] = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        self.assertEqual(ScalarMin._reset_calls_counter, 0)
        res1 = stirrer.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def test_repair_min4(self):
        """
        test_repair_min4()
        runs with sensitive ids update
        """
        s=Scheduler()
        ScalarMin._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        min_=ScalarMin(name='min_repair_test4', scheduler=s)
        stirrer = MyStirrer(watched='min_repair_test4', mode='update', value=-9999.0, scheduler=s)
        stirrer.input[0] = random.output.result
        min_.input[0] = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        self.assertEqual(ScalarMin._reset_calls_counter, 0)
        res1 = stirrer.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def test_repair_min5(self):
        """
        test_repair_min5()
        runs with sensitive ids update (critical)
        """
        s=Scheduler()
        ScalarMin._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        min_=ScalarMin(name='min_repair_test4', scheduler=s)
        stirrer = MyStirrer(watched='min_repair_test4', mode='update', value=9999.0, scheduler=s)
        stirrer.input[0] = random.output.result
        min_.input[0] = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        self.assertEqual(ScalarMin._reset_calls_counter, 1)
        res1 = stirrer.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def test_repair_min6(self):
        """
        test_repair_min6()
        runs with NON-sensitive ids updates
        """
        s=Scheduler()
        ScalarMin._reset_calls_counter = 0
        random = RandomTable(2, rows=100000, scheduler=s)
        min_=ScalarMin(name='min_repair_test5', scheduler=s)
        stirrer = MyStirrer(watched='min_repair_test5', proc_sensitive=False,
                            mode='update', scheduler=s)
        stirrer.input[0] = random.output.result
        min_.input[0] = stirrer.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        self.assertEqual(ScalarMin._reset_calls_counter, 0)
        res1 = stirrer.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def compare(self, res1, res2):
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        #print('v1 = ', v1, res1.keys())
        #print('v2 = ', v2, res2.keys())
        self.assertTrue(np.allclose(v1, v2))

if __name__ == '__main__':
    unittest.main()
