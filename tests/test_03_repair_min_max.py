from __future__ import annotations

from . import ProgressiveTest
from progressivis import Print, Scheduler
from progressivis.core.module import Module, def_input, def_output
from progressivis.table.table import PTable
from progressivis.stats import RandomPTable, ScalarMax, ScalarMin
from progressivis.core.pintset import PIntSet
from progressivis.core import aio
from progressivis.core.utils import indices_len, fix_loc
import numpy as np

from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.module import ReturnRunStep

ScalarMax._reset_calls_counter = 0  # type: ignore
ScalarMax._orig_reset = ScalarMax.reset  # type: ignore


def _reset_func_max(self_: ScalarMax) -> None:
    ScalarMax._reset_calls_counter += 1  # type: ignore
    return ScalarMax._orig_reset(self_)  # type: ignore


ScalarMax.reset = _reset_func_max  # type: ignore

ScalarMin._reset_calls_counter = 0  # type: ignore
ScalarMin._orig_reset = ScalarMin.reset  # type: ignore


def _reset_func_min(self_: ScalarMin) -> None:
    ScalarMin._reset_calls_counter += 1  # type: ignore
    return ScalarMin._orig_reset(self_)  # type: ignore


ScalarMin.reset = _reset_func_min  # type: ignore


@def_input("table", PTable)
@def_output("result", PTable)
class MyStirrer(Module):
    def __init__(
        self,
        watched: str,
        proc_sensitive: bool = True,
        mode: str = "delete",
        value: float = 9999.0,
        **kwds: Any,
    ):
        super().__init__(**kwds)
        self.watched = watched
        self.proc_sensitive = proc_sensitive
        self.mode = mode
        self.default_step_size = 100
        self.value = value
        self.done = False

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        # input_slot.update(run_number)
        steps = 0
        if not input_slot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)
        created = input_slot.created.next(step_size)
        steps = indices_len(created)
        input_table = input_slot.data()
        if self.result is None:
            self.result = PTable(
                self.generate_table_name("stirrer"),
                dshape=input_table.dshape,
            )
        v = input_table.loc[fix_loc(created), :]
        self.result.append(v)
        if not self.done:
            module = self.scheduler()[self.watched]
            sensitive_ids = PIntSet(getattr(module, "_sensitive_ids").values())
            if sensitive_ids:
                if self.proc_sensitive:
                    if self.mode == "delete":
                        # print('delete sensitive', sensitive_ids)
                        del self.result.loc[sensitive_ids]
                    else:
                        # print('update sensitive', sensitive_ids)
                        self.result.loc[sensitive_ids, 0] = self.value
                    self.done = True
                else:  # non sensitive
                    if len(self.result) > 10:
                        for i in range(10):
                            id_ = self.result.index[i]
                            if id_ not in sensitive_ids:
                                if self.mode == "delete":
                                    del self.result.loc[id_]
                                else:
                                    self.result.loc[id_, 0] = self.value
                                self.done = True

        return self._return_run_step(self.next_state(input_slot), steps_run=steps)


# @skip
class TestRepairMax(ProgressiveTest):
    def test_repair_max(self) -> None:
        """
        test_repair_max()
        max without deletes/updates
        """
        s = Scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        max_ = ScalarMax(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert random.result is not None
        assert max_.result is not None
        res1 = random.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_repair_max2(self) -> None:
        """
        test_repair_max2()
        runs with sensitive ids deletion
        """
        s = Scheduler()
        ScalarMax._reset_calls_counter = 0  # type: ignore
        random = RandomPTable(2, rows=100000, scheduler=s)
        max_ = ScalarMax(name="max_repair_test2", scheduler=s)
        stirrer = MyStirrer(watched="max_repair_test2", scheduler=s)
        stirrer.input[0] = random.output.result
        max_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert max_.result is not None
        self.assertEqual(ScalarMax._reset_calls_counter, 1)  # type: ignore
        res1 = stirrer.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_repair_max3(self) -> None:
        """
        test_repair_max3()
        runs with NON-sensitive ids deletion
        """
        s = Scheduler()
        ScalarMax._reset_calls_counter = 0  # type: ignore
        random = RandomPTable(2, rows=100000, scheduler=s)
        max_ = ScalarMax(name="max_repair_test3", scheduler=s)
        stirrer = MyStirrer(
            watched="max_repair_test3", proc_sensitive=False, scheduler=s
        )
        stirrer.input[0] = random.output.result
        max_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert max_.result is not None
        self.assertEqual(ScalarMax._reset_calls_counter, 0)  # type: ignore
        res1 = stirrer.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_repair_max4(self) -> None:
        """
        test_repair_max4()
        runs with sensitive ids update
        """
        s = Scheduler()
        ScalarMax._reset_calls_counter = 0  # type: ignore
        random = RandomPTable(2, rows=100000, scheduler=s)
        max_ = ScalarMax(name="max_repair_test4", scheduler=s)
        stirrer = MyStirrer(
            watched="max_repair_test4", mode="update", value=9999.0, scheduler=s
        )
        stirrer.input[0] = random.output.result
        max_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert max_.result is not None
        self.assertEqual(ScalarMax._reset_calls_counter, 0)  # type: ignore
        res1 = stirrer.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_repair_max5(self) -> None:
        """
        test_repair_max5()
        runs with sensitive ids update (critical)
        """
        s = Scheduler()
        ScalarMax._reset_calls_counter = 0  # type: ignore
        random = RandomPTable(2, rows=100000, scheduler=s)
        max_ = ScalarMax(name="max_repair_test4", scheduler=s)
        stirrer = MyStirrer(
            watched="max_repair_test4", mode="update", value=-9999.0, scheduler=s
        )
        stirrer.input[0] = random.output.result
        max_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert max_.result is not None
        self.assertEqual(ScalarMax._reset_calls_counter, 1)  # type: ignore
        res1 = stirrer.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_repair_max6(self) -> None:
        """
        test_repair_max6()
        runs with NON-sensitive ids updates
        """
        s = Scheduler()
        ScalarMax._reset_calls_counter = 0  # type: ignore
        random = RandomPTable(2, rows=100000, scheduler=s)
        max_ = ScalarMax(name="max_repair_test5", scheduler=s)
        stirrer = MyStirrer(
            watched="max_repair_test5", proc_sensitive=False, mode="update", scheduler=s
        )
        stirrer.input[0] = random.output.result
        max_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert max_.result is not None
        self.assertEqual(ScalarMax._reset_calls_counter, 0)  # type: ignore
        res1 = stirrer.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def compare(self, res1: Dict[str, Any], res2: Dict[str, Any]) -> None:
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        # print('v1 = ', v1, res1.keys())
        # print('v2 = ', v2, res2.keys())
        self.assertTrue(np.allclose(v1, v2))


class TestRepairMin(ProgressiveTest):
    def test_repair_min(self) -> None:
        """
        test_repair_min()
        min without deletes/updates
        """
        s = Scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        min_ = ScalarMin(name="min_" + str(hash(random)), scheduler=s)
        min_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        assert random.result is not None
        assert min_.result is not None
        res1 = random.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def test_repair_min2(self) -> None:
        """
        test_repair_min2()
        runs with sensitive ids deletion
        """
        s = Scheduler()
        ScalarMin._reset_calls_counter = 0  # type: ignore
        random = RandomPTable(2, rows=100000, scheduler=s)
        min_ = ScalarMin(name="min_repair_test2", scheduler=s)
        stirrer = MyStirrer(watched="min_repair_test2", scheduler=s)
        stirrer.input[0] = random.output.result
        min_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert min_.result is not None
        self.assertEqual(ScalarMin._reset_calls_counter, 1)  # type: ignore
        res1 = stirrer.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def test_repair_min3(self) -> None:
        """
        test_repair_min3()
        runs with NON-sensitive ids deletion
        """
        s = Scheduler()
        ScalarMin._reset_calls_counter = 0  # type: ignore
        random = RandomPTable(2, rows=100000, scheduler=s)
        min_ = ScalarMin(name="min_repair_test3", scheduler=s)
        stirrer = MyStirrer(
            watched="min_repair_test3", proc_sensitive=False, scheduler=s
        )
        stirrer.input[0] = random.output.result
        min_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert min_.result is not None
        self.assertEqual(ScalarMin._reset_calls_counter, 0)  # type: ignore
        res1 = stirrer.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def test_repair_min4(self) -> None:
        """
        test_repair_min4()
        runs with sensitive ids update
        """
        s = Scheduler()
        ScalarMin._reset_calls_counter = 0  # type: ignore
        random = RandomPTable(2, rows=100000, scheduler=s)
        min_ = ScalarMin(name="min_repair_test4", scheduler=s)
        stirrer = MyStirrer(
            watched="min_repair_test4", mode="update", value=-9999.0, scheduler=s
        )
        stirrer.input[0] = random.output.result
        min_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert min_.result is not None
        self.assertEqual(ScalarMin._reset_calls_counter, 0)  # type: ignore
        res1 = stirrer.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def test_repair_min5(self) -> None:
        """
        test_repair_min5()
        runs with sensitive ids update (critical)
        """
        s = Scheduler()
        ScalarMin._reset_calls_counter = 0  # type: ignore
        random = RandomPTable(2, rows=100000, scheduler=s)
        min_ = ScalarMin(name="min_repair_test4", scheduler=s)
        stirrer = MyStirrer(
            watched="min_repair_test4", mode="update", value=9999.0, scheduler=s
        )
        stirrer.input[0] = random.output.result
        min_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert min_.result is not None
        self.assertEqual(ScalarMin._reset_calls_counter, 1)  # type: ignore
        res1 = stirrer.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def test_repair_min6(self) -> None:
        """
        test_repair_min6()
        runs with NON-sensitive ids updates
        """
        s = Scheduler()
        ScalarMin._reset_calls_counter = 0  # type: ignore
        random = RandomPTable(2, rows=100000, scheduler=s)
        min_ = ScalarMin(name="min_repair_test5", scheduler=s)
        stirrer = MyStirrer(
            watched="min_repair_test5", proc_sensitive=False, mode="update", scheduler=s
        )
        stirrer.input[0] = random.output.result
        min_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = min_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        assert min_.result is not None
        self.assertEqual(ScalarMin._reset_calls_counter, 0)  # type: ignore
        res1 = stirrer.result.min()
        res2 = min_.result
        self.compare(res1, res2)

    def compare(self, res1: Dict[str, Any], res2: Dict[str, Any]) -> None:
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        self.assertTrue(np.allclose(v1, v2))


if __name__ == "__main__":
    ProgressiveTest.main()
