"Test for SlotHub"
from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Print
from progressivis.stats import Min, Max, RandomPTable
from progressivis.core.module_facade import ModuleFacade
from progressivis.table.table_facade import TableFacade

import numpy as np

from typing import Any, Dict


class TestModuleFacade(ProgressiveTest):
    def test_module_facade(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10000, scheduler=s)
        min_ = Min(name="min_" + str(hash(random)), scheduler=s)
        min_.input[0] = random.output.result
        max_ = Max(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = random.output.result
        hub = ModuleFacade()
        hub.add_proxy("min", "result", min_)
        hub.add_proxy("max", "result", max_)
        pr_min = Print(scheduler=s)
        pr_min.input[0] = hub.output.min
        pr_max = Print(scheduler=s)
        pr_max.input[0] = hub.output.max
        aio.run(s.start())
        assert random.result is not None
        assert min_.result is not None
        assert max_.result is not None
        res1 = random.result.min()
        res2 = min_.result
        self.compare(res1, res2)
        res1 = random.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_table_module(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10000, scheduler=s)
        tabmod = TableFacade.get_or_create(random, "result")
        pr_min = Print(scheduler=s)
        pr_min.input[0] = tabmod.output.min
        pr_max = Print(scheduler=s)
        pr_max.input[0] = tabmod.output.max
        aio.run(s.start())

    def compare(self, res1: Dict[str, Any], res2: Dict[str, Any]) -> None:
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        self.assertTrue(np.allclose(v1, v2))


if __name__ == "__main__":
    ProgressiveTest.main()
