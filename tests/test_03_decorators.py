# type: ignore
from . import ProgressiveTest
from progressivis import (
    Scheduler,
    CSVLoader,
    Constant,
    Module,
    ReturnRunStep,
    def_input,
    def_output,
    PTable
)
from progressivis.core.api import (
    run_if_all,
    run_always,
    or_all,
    run_if_any,
    and_any,
)
from progressivis.datasets import get_dataset
from progressivis.core.decorators import _CtxImpl, process_slot
import asyncio as aio

from typing import Any, Callable, Coroutine


@def_input("a", PTable)
@def_input("b", PTable)
@def_input("c", PTable)
@def_input("d", PTable)
@def_output("result", PTable)
class FooABC(Module):
    def __init__(self, **kwds: Any) -> None:
        super().__init__(output_required=False, **kwds)

    def run_step_impl(
        self, ctx: _CtxImpl, run_number: int, step_size: int
    ) -> ReturnRunStep:
        if self.result is None:
            self.result = PTable(
                self.generate_table_name("Foo"), dshape="{a: int, b: int}", create=True
            )
        for sn in "abcd":
            getattr(ctx, sn).created.next()
        self.result.append({"a": [run_number], "b": [step_size]})
        return self._return_run_step(self.state_blocked, steps_run=0)


class RunIfAll(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)
    @run_if_all
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)


class RunAlways(FooABC):
    def is_greedy(self) -> bool:
        return True

    @process_slot("a", "b", "c", "d", reset_if=False)
    @run_always
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)


class RunIfAllacOrAllbd(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)  # type: ignore
    @run_if_all("a", "c")  # type: ignore
    @or_all("b", "d")  # type: ignore
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            assert self.context
            return self.run_step_impl(ctx, run_number, step_size)


class RunIfAllabOrAllcd(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)  # type: ignore
    @run_if_all("a", "b")  # type: ignore
    @or_all("c", "d")  # type: ignore
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)


class RunIfAny(FooABC):
    def is_greedy(self) -> bool:
        return True

    @process_slot("a", "b", "c", "d", reset_if=False)  # type: ignore
    @run_if_any()  # type: ignore
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context is not None
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)


class RunIfAnyAndAny(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)  # type: ignore
    @run_if_any("a", "c")  # type: ignore
    @and_any("b", "d")  # type: ignore
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context is not None
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)


class InvalidProcessAfterRun(FooABC):
    @run_if_any("a", "c")  # type: ignore
    @process_slot("a", "b", "c", "d", reset_if=False)  # type: ignore
    @and_any("b", "d")  # type: ignore
    def run_step(
            self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context is not None
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)


class InvalidDoubleRun(FooABC):
    @process_slot("a", "b", "c", "d", reset_if=False)  # type: ignore
    @run_if_any("a", "c")  # type: ignore
    @run_if_any("b", "d")  # type: ignore
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context is not None
        with self.context as ctx:
            return self.run_step_impl(ctx, run_number, step_size)


def _4_csv_scenario(
    module: Module, s: Scheduler
) -> Callable[[Scheduler, int], Coroutine[Any, Any, None]]:
    with s:
        csv_a = CSVLoader(
            get_dataset("smallfile"), header=None, scheduler=s
        )
        csv_b = CSVLoader(
            get_dataset("smallfile"), header=None, scheduler=s
        )
        csv_c = CSVLoader(
            get_dataset("smallfile"), header=None, scheduler=s
        )
        csv_d = CSVLoader(
            get_dataset("smallfile"), header=None, scheduler=s
        )
        module.input.a = csv_a.output.result
        module.input.b = csv_b.output.result
        module.input.c = csv_c.output.result
        module.input.d = csv_d.output.result

    async def _fun(s: Scheduler, r: int) -> None:
        if r > 10:
            s.task_stop()

    return _fun


def _4_const_scenario(module: Module, s: Scheduler) -> Callable[[Scheduler, int], None]:
    table_ = PTable("const_4_scenario", dshape="{a: int}", create=True)
    const_a = Constant(table=table_, scheduler=s)
    const_b = Constant(table=table_, scheduler=s)
    const_c = Constant(table=table_, scheduler=s)
    const_d = Constant(table=table_, scheduler=s)
    module.input.a = const_a.output.result
    module.input.b = const_b.output.result
    module.input.c = const_c.output.result
    module.input.d = const_d.output.result

    def _fun(s: Scheduler, r: int) -> None:
        if r > 10:
            s.task_stop()

    return _fun


def _2_csv_2_const_scenario(
    module: Module, s: Scheduler
) -> Callable[[Scheduler, int], None]:
    csv_a = CSVLoader(
        get_dataset("smallfile"), header=None, scheduler=s
    )
    csv_b = CSVLoader(
        get_dataset("smallfile"), header=None, scheduler=s
    )
    table_c = PTable("const_c_2_csv_2_const_scenario", dshape="{a: int}", create=True)
    const_c = Constant(table=table_c, scheduler=s)
    table_d = PTable("const_d_2_csv_2_const_scenario", dshape="{a: int}", create=True)
    const_d = Constant(table=table_d, scheduler=s)
    module.input.a = csv_a.output.result
    module.input.b = csv_b.output.result
    module.input.c = const_c.output.result
    module.input.d = const_d.output.result

    def _fun(s: Scheduler, r: int) -> None:
        if r > 10:
            s.task_stop()

    return _fun


# @skip
class TestDecorators(ProgressiveTest):
    def test_decorators_all(self) -> None:
        s = self.scheduler()
        module = RunIfAll(scheduler=s)
        _fun = _4_csv_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            module.result is not None
        )  # evidence that run_step_impl() was called
        assert module.context is not None
        self.assertEqual(module.context._slot_policy, "run_if_all")
        self.assertEqual(module.context._slot_expr, [["a", "b", "c", "d"]])

    def test_decorators_all_or_all(self) -> None:
        s = self.scheduler()
        module = RunIfAllacOrAllbd(scheduler=s)
        _fun = _4_csv_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            module.result is not None
        )  # evidence that run_step_impl() was called
        assert module.context is not None
        self.assertEqual(module.context._slot_policy, "run_if_all")
        self.assertEqual(module.context._slot_expr, [("a", "c"), ("b", "d")])

    def test_decorators_any(self) -> None:
        s = self.scheduler()
        module = RunIfAny(scheduler=s)
        _fun = _4_csv_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            module.result is not None
        )  # evidence that run_step_impl() was called
        assert module.context is not None
        self.assertEqual(module.context._slot_policy, "run_if_any")
        self.assertEqual(module.context._slot_expr, [["a", "b", "c", "d"]])

    def test_decorators_any_and_any(self) -> None:
        s = self.scheduler()
        module = RunIfAnyAndAny(scheduler=s)
        _fun = _4_csv_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            module.result is not None
        )  # evidence that run_step_impl() was called
        assert module.context is not None
        self.assertEqual(module.context._slot_policy, "run_if_any")
        self.assertEqual(module.context._slot_expr, [("a", "c"), ("b", "d")])


# @skip
class TestDecoratorsWith2CSV2Const(ProgressiveTest):
    def test_decorators_all(self) -> None:
        s = self.scheduler()
        module = RunIfAll(scheduler=s)
        _fun = _2_csv_2_const_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            module.result is None
        )  # evidence that run_step_impl() was NOT called
        assert module.context is not None
        self.assertEqual(module.context._slot_policy, "run_if_all")
        self.assertEqual(module.context._slot_expr, [["a", "b", "c", "d"]])

    def test_decorators_all_or_all(self) -> None:
        s = self.scheduler()
        module = RunIfAllacOrAllbd(scheduler=s)
        _fun = _2_csv_2_const_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            module.result is None
        )  # evidence that run_step_impl() was NOT called
        assert module.context is not None
        self.assertEqual(module.context._slot_policy, "run_if_all")
        self.assertEqual(module.context._slot_expr, [("a", "c"), ("b", "d")])

    def test_decorators_all_or_all2(self) -> None:
        s = self.scheduler()
        module = RunIfAllabOrAllcd(scheduler=s)
        _fun = _2_csv_2_const_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            module.result is not None
        )  # evidence that run_step_impl() was called
        assert module.context is not None
        self.assertEqual(module.context._slot_policy, "run_if_all")
        self.assertEqual(module.context._slot_expr, [("a", "b"), ("c", "d")])


class TestDecoratorsWith2CSV2Const2(ProgressiveTest):
    def test_decorators_any(self) -> None:
        s = self.scheduler()
        module = RunIfAny(scheduler=s)
        _fun = _2_csv_2_const_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            module.result is not None
        )  # evidence that run_step_impl() was called
        assert module.context is not None
        self.assertEqual(module.context._slot_policy, "run_if_any")
        self.assertEqual(module.context._slot_expr, [["a", "b", "c", "d"]])

    def test_decorators_any_and_any(self) -> None:
        s = self.scheduler()
        module = RunIfAnyAndAny(scheduler=s)
        _fun = _4_csv_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            module.result is not None
        )  # evidence that run_step_impl() was called
        assert module.context is not None
        self.assertEqual(module.context._slot_policy, "run_if_any")
        self.assertEqual(module.context._slot_expr, [("a", "c"), ("b", "d")])


# @skip
class TestDecoratorsWith4Const(ProgressiveTest):
    def test_decorators_any(self) -> None:
        s = self.scheduler()
        module = RunIfAny(scheduler=s)
        _fun = _4_const_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            module.result is None
        )  # evidence that run_step_impl() was NOT called
        assert module.context is not None
        self.assertEqual(module.context._slot_policy, "run_if_any")
        self.assertEqual(module.context._slot_expr, [["a", "b", "c", "d"]])

    def test_decorators_always(self) -> None:
        s = self.scheduler()
        module = RunAlways(scheduler=s)
        _fun = _4_const_scenario(module, s)
        aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            module.result is not None
        )  # evidence that run_step_impl() was called despite slots inactivity
        assert module.context is not None
        self.assertEqual(module.context._slot_policy, "run_always")
        self.assertEqual(module.context._slot_expr, [["a", "b", "c", "d"]])


# @skip
class TestDecoratorsInvalid(ProgressiveTest):
    def test_invalid_process_after_run(self) -> None:
        with self.assertRaises(RuntimeError) as cm:
            s = self.scheduler()
            module = InvalidProcessAfterRun(scheduler=s)
            _fun = _4_csv_scenario(module, s)
            aio.run(s.start(tick_proc=_fun))
        self.assertTrue(
            "context not found. consider processing slots before"
            in cm.exception.args[0]
        )

    def test_invalid_double_run(self) -> None:
        with self.assertRaises(RuntimeError) as cm:
            s = self.scheduler()
            module = InvalidDoubleRun(scheduler=s)
            _fun = _4_csv_scenario(module, s)
            aio.run(s.start(tick_proc=_fun))
        self.assertTrue("run_if_any cannot follow run_if_any" in cm.exception.args[0])


"""
+class TestDecoratorsDelModule(ProgressiveTest):
+    def test_decorators_del_module(self):
+        s = self.scheduler()
+        with s:
+        #if True:
+            module = RunIfAllabOrAllcd(scheduler=s)
+            _fun = _2_csv_2_const_scenario(module, s)
+            async def _remove_module(sch):
+                await aio.sleep(1)
+                print("AWAKED!")
+                import pdb;pdb.set_trace()
+                #m = s._modules['constant_1']
+                #s.dataflow.remove_module(m)
+                del sch['constant_1']
+            #s.start()
+            async def _gather(s):
+                await aio.gather(s.start(tick_proc=_fun), _remove_module(s) )
+            aio.run(_gather(s))
+
"""

if __name__ == "__main__":
    ProgressiveTest.main()
