from . import ProgressiveTest
from progressivis import Print, Scheduler, RandomPTable, Max
from progressivis.table.stirrer import Stirrer
from progressivis.table.switch import Switch
from progressivis.core import aio
from typing import Any, Dict
import numpy as np


class TestSwitch(ProgressiveTest):
    def test_switch_if_then(self) -> None:
        s = Scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(
            update_column="_1",
            delete_rows=5,
            update_rows=5,
            fixed_step_size=100,
            scheduler=s,
        )
        stirrer.input[0] = random.output.result
        switch = Switch(condition=lambda x: True, scheduler=s)
        switch.input[0] = stirrer.output.result
        max_ = Max(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = switch.output.result
        pr_else = Print(proc=self.terse, scheduler=s)
        pr_else.input[0] = switch.output.result_else
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        res1 = stirrer.result.max()
        res2 = max_.result
        assert res2 is not None
        self.compare(res1, res2)

    def test_switch_if_else(self) -> None:
        s = Scheduler()
        random = RandomPTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(
            update_column="_1",
            delete_rows=5,
            update_rows=5,
            fixed_step_size=100,
            scheduler=s,
        )
        stirrer.input[0] = random.output.result
        switch = Switch(condition=lambda x: False, scheduler=s)
        switch.input[0] = stirrer.output.result
        max_ = Max(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = switch.output.result_else
        pr_then = Print(proc=self.terse, scheduler=s)
        pr_then.input[0] = switch.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        res1 = stirrer.result.max()
        assert max_.result is not None
        res2 = max_.result
        self.compare(res1, res2)

    def compare(self, res1: Dict[Any, Any], res2: Dict[Any, Any]) -> None:
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        self.assertEqual(v1.shape, v2.shape)
        # print('v1 = ', v1)
        # print('v2 = ', v2)
        self.assertTrue(np.allclose(v1, v2))


if __name__ == "__main__":
    ProgressiveTest.main()
