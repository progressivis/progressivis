from . import ProgressiveTest
from progressivis import Print, Scheduler
from progressivis.stats import RandomPTable, Max, Min
from progressivis.table.stirrer import Stirrer
from progressivis.table.switch import Switch
from progressivis.table.hub import Hub
from progressivis.core import aio

import numpy as np


class TestHub(ProgressiveTest):
    def test_hub_if_then(self):
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
        min_ = Min(name="min_" + str(hash(random)), scheduler=s)
        min_.input[0] = switch.output.result_else
        hub = Hub(scheduler=s)
        hub.input.table = min_.output.result
        hub.input.table = max_.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = hub.output.result
        aio.run(s.start())
        res1 = stirrer.result.max()
        res2 = hub.result
        self.compare(res1, res2)

    def test_hub_if_else(self):
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
        max_.input[0] = switch.output.result
        min_ = Min(name="min_" + str(hash(random)), scheduler=s)
        min_.input[0] = switch.output.result_else
        hub = Hub(scheduler=s)
        hub.input.table = min_.output.result
        hub.input.table = max_.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = hub.output.result
        aio.run(s.start())
        res1 = stirrer.result.min()
        res2 = hub.result
        self.compare(res1, res2)

    def compare(self, res1, res2):
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        self.assertEqual(v1.shape, v2.shape)
        # print('v1 = ', v1)
        # print('v2 = ', v2)
        self.assertTrue(np.allclose(v1, v2))


if __name__ == "__main__":
    ProgressiveTest.main()
