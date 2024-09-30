from . import ProgressiveTest, skipIf
from progressivis.core import aio
from progressivis import Print, RandomPTable
from progressivis.table.stirrer import Stirrer, StirrerView
from progressivis.stats.cxxmax import Max, CxxMax  # type: ignore

import numpy as np

from typing import Dict, Any

# CxxMax = None  # Skip for now


class TestCxxMax(ProgressiveTest):
    def compare(self, res1: Dict[str, Any], res2: Dict[str, Any]) -> None:
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        # print('v1 = ', v1)
        # print('v2 = ', v2)
        self.assertTrue(np.allclose(v1, v2))

    @skipIf(CxxMax is None, "C++ module is missing")
    def test_max(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10_000, scheduler=s)
        max_ = Max(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert random.result is not None
        res1 = random.result.max()
        res2 = max_.cxx_module.get_output_table().last().to_dict(ordered=True)
        self.compare(res1, res2)

    @skipIf(CxxMax is None, "C++ module is missing")
    def test_stirrer(self) -> None:
        s = self.scheduler()
        random = RandomPTable(2, rows=100_000, scheduler=s)
        stirrer = Stirrer(
            update_column="_1",
            delete_rows=5,
            update_rows=5,
            fixed_step_size=100,
            scheduler=s,
        )
        stirrer.input[0] = random.output.result
        max_ = Max(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert random.result is not None
        res1 = random.result.max()
        res2 = max_.cxx_module.get_output_table().last().to_dict(ordered=True)
        self.compare(res1, res2)

    @skipIf(CxxMax is None, "C++ module is missing")
    def test_stirrer_view(self) -> None:
        s = self.scheduler()
        random = RandomPTable(2, rows=100_000, scheduler=s)
        stirrer = StirrerView(
            update_column="_1",
            delete_rows=5,
            update_rows=5,
            fixed_step_size=100,
            scheduler=s,
        )
        stirrer.input[0] = random.output.result
        max_ = Max(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert random.result is not None
        res1 = random.result.max()
        res2 = max_.cxx_module.get_output_table().last().to_dict(ordered=True)
        self.compare(res1, res2)


if __name__ == "__main__":
    ProgressiveTest.main()
