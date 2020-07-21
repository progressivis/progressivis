from . import ProgressiveTest, skip, skipIf
from progressivis import Every
from progressivis.stats import Percentiles
from progressivis.io import CSVLoader
from progressivis.table.constant import Constant
from progressivis.core.slot import SlotDescriptor
from progressivis.datasets import get_dataset
from progressivis.table.module import TableModule
from progressivis.table.table import Table
from progressivis.core.decorators import *
import asyncio as aio

class FooABC(TableModule):
    def __init__(self, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('a', type=Table, required=True),
                         SlotDescriptor('b', type=Table, required=True),
                         SlotDescriptor('c', type=Table, required=True),
                         SlotDescriptor('d', type=Table, required=True),
                        ])
        super().__init__(**kwds)
    def run_step_impl(self, ctx, run_number, step_size):
        if self._table is None:
            self._table = Table(self.generate_table_name('Foo'),
                                dshape="{a: int, b: int}", create=True)
        for sn in "abcd":
            getattr(ctx, sn).created.next()
        self._table.append({'a': [run_number], 'b': [step_size]})
        return self._return_run_step(self.state_blocked, steps_run=0)
        
class RunIfAll(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)    
    @run_if_all
    @check_slots
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)

class RunIfAllacOrAllbd(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)    
    @run_if_all('a', 'c')
    @or_all('b', 'd')
    @check_slots
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)

class RunIfAllabOrAllcd(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)    
    @run_if_all('a', 'b')
    @or_all('c', 'd')
    @check_slots
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)

class RunIfAny(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)    
    @run_if_any()
    @check_slots
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)

class RunIfAnyAndAny(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)    
    @run_if_any('a', 'c')
    @and_any('b', 'd')
    @check_slots
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)

class InvalidNoCheck(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)    
    @run_if_any('a', 'c')
    @and_any('b', 'd')
    #@check_slots
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)

class InvalidDoubleRun(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)    
    @run_if_any('a', 'c')
    @run_if_any('b', 'd')
    @check_slots
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)

def _4_csv_scenario(module, s):
        csv_a = CSVLoader(get_dataset('bigfile'), index_col=False,
                               header=None, scheduler=s)
        csv_b = CSVLoader(get_dataset('bigfile'), index_col=False,
                               header=None, scheduler=s)
        csv_c = CSVLoader(get_dataset('bigfile'), index_col=False,
                               header=None, scheduler=s)
        csv_d = CSVLoader(get_dataset('bigfile'), index_col=False,
                               header=None, scheduler=s)
        module.input.a = csv_a.output.table
        module.input.b = csv_b.output.table
        module.input.c = csv_c.output.table
        module.input.d = csv_d.output.table        
        def _fun(s,r):
            if r>10:
                s.task_stop()
        return _fun
    
def _2_csv_2_const_scenario(module, s):
        csv_a = CSVLoader(get_dataset('bigfile'), index_col=False,
                               header=None, scheduler=s)
        csv_b = CSVLoader(get_dataset('bigfile'), index_col=False,
                               header=None, scheduler=s)
        table_c = Table('const_c_2_csv_2_const_scenario', dshape="{a: int}", create=True)
        const_c = Constant(table=table_c, scheduler=s)
        table_d = Table('const_d_2_csv_2_const_scenario', dshape="{a: int}", create=True)
        const_d = Constant(table=table_d, scheduler=s)
        module.input.a = csv_a.output.table
        module.input.b = csv_b.output.table
        module.input.c = const_c.output.table
        module.input.d = const_d.output.table        
        def _fun(s,r):
            if r>10:
                s.task_stop()
        return _fun
#@skip    
class TestDecorators(ProgressiveTest):
    def test_decorators_all(self):
        s = self.scheduler()
        module = RunIfAll(scheduler=s)
        _fun = _4_csv_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(module.table() is not None) # evidence that run_step_impl() was called
        self.assertEqual(module.context._slot_policy, 'run_if_all')
        self.assertEqual(module.context._slot_expr, [['a', 'b', 'c', 'd']])

    def test_decorators_all_or_all(self):
        s = self.scheduler()
        module = RunIfAllacOrAllbd(scheduler=s)
        _fun = _4_csv_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(module.table() is not None) # evidence that run_step_impl() was called
        self.assertEqual(module.context._slot_policy, 'run_if_all')
        self.assertEqual(module.context._slot_expr, [('a', 'c'), ('b', 'd')])

    def test_decorators_any(self):
        s = self.scheduler()
        module = RunIfAny(scheduler=s)
        _fun = _4_csv_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(module.table() is not None) # evidence that run_step_impl() was called
        self.assertEqual(module.context._slot_policy, 'run_if_any')
        self.assertEqual(module.context._slot_expr, [['a', 'b', 'c', 'd']])

    def test_decorators_any_and_any(self):
        s = self.scheduler()
        module = RunIfAnyAndAny(scheduler=s)
        _fun = _4_csv_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(module.table() is not None) # evidence that run_step_impl() was called
        self.assertEqual(module.context._slot_policy, 'run_if_any')
        self.assertEqual(module.context._slot_expr, [('a', 'c'), ('b', 'd')])
#@skip
class TestDecoratorsWithConst(ProgressiveTest):
    def test_decorators_all(self):
        s = self.scheduler()
        module = RunIfAll(scheduler=s)
        _fun = _2_csv_2_const_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(module.table() is None) # evidence that run_step_impl() was NOT called
        self.assertEqual(module.context._slot_policy, 'run_if_all')
        self.assertEqual(module.context._slot_expr, [['a', 'b', 'c', 'd']])

    def test_decorators_all_or_all(self):
        s = self.scheduler()
        module = RunIfAllacOrAllbd(scheduler=s)
        _fun = _2_csv_2_const_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(module.table() is None) # evidence that run_step_impl() was NOT called
        self.assertEqual(module.context._slot_policy, 'run_if_all')
        self.assertEqual(module.context._slot_expr, [('a', 'c'), ('b', 'd')])

    def test_decorators_all_or_all2(self):
        s = self.scheduler()
        module = RunIfAllabOrAllcd(scheduler=s)
        _fun = _2_csv_2_const_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(module.table() is not None) # evidence that run_step_impl() was called
        self.assertEqual(module.context._slot_policy, 'run_if_all')
        self.assertEqual(module.context._slot_expr, [('a', 'b'), ('c', 'd')])

    def test_decorators_any(self):
        s = self.scheduler()
        module = RunIfAny(scheduler=s)
        _fun = _2_csv_2_const_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(module.table() is not None) # evidence that run_step_impl() was called
        self.assertEqual(module.context._slot_policy, 'run_if_any')
        self.assertEqual(module.context._slot_expr, [['a', 'b', 'c', 'd']])

    def test_decorators_any_and_any(self):
        s = self.scheduler()
        module = RunIfAnyAndAny(scheduler=s)
        _fun = _4_csv_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(module.table() is not None) # evidence that run_step_impl() was called
        self.assertEqual(module.context._slot_policy, 'run_if_any')
        self.assertEqual(module.context._slot_expr, [('a', 'c'), ('b', 'd')])
#@skip
class TestDecoratorsInvalid(ProgressiveTest):
    def test_invalid_no_check(self):
        with self.assertRaises(RuntimeError) as cm:
            s = self.scheduler()
            module = InvalidNoCheck(scheduler=s)
            _fun = _4_csv_scenario(module, s)
            aio.run(s.start(tick_proc=_fun))
        self.assertTrue('check_slots decorator is missing' in cm.exception.args[0])

    def test_invalid_double_run(self):
        with self.assertRaises(RuntimeError) as cm:
            s = self.scheduler()
            module = InvalidDoubleRun(scheduler=s)
            _fun = _4_csv_scenario(module, s)
            aio.run(s.start(tick_proc=_fun))
        self.assertTrue('run_if_any cannot follow run_if_any' in cm.exception.args[0])

if __name__ == '__main__':
    ProgressiveTest.main()
