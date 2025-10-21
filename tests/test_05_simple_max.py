"""
Test the SimpleMax Module from the doc example
"""
from . import ProgressiveTest

from progressivis.core import aio
from progressivis import Tick, RandomPTable
from progressivis.table.stirrer import Stirrer

import numpy as np

from typing import Any, Dict

import os
import sys


script_dir = os.path.dirname(__file__)
doc_dir = os.path.join(script_dir, '..', 'doc')
sys.path.append(doc_dir)

from simple_max import SimpleMax  # noqa: E402


Max = SimpleMax


class TestSimpleMax(ProgressiveTest):
    def compare(self, res1: Dict[str, Any], res2: Dict[str, Any]) -> None:
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        self.assertTrue(np.allclose(v1, v2))

    def test_max(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=10000, scheduler=s)
        max_ = Max(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = random.output.result
        pr = Tick(scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert random.result is not None
        assert max_.result is not None
        res1 = random.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def test_max_reset(self) -> None:
        s = self.scheduler
        random = RandomPTable(10, rows=10000, scheduler=s)
        stirrer = Stirrer(
            update_column="_2",
            fixed_step_size=1000,
            scheduler=s,
            delete_rows=5
        )
        stirrer.input[0] = random.output.result
        max_ = Max(scheduler=s)
        max_.input[0] = stirrer.output.result
        pr = Tick(scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        assert stirrer.result is not None
        tab = stirrer.result.loc[:, ["_2"]]
        assert tab is not None
        v = tab.to_array().reshape(-1)
        res1 = v.max()
        assert max_.result is not None
        res2 = max_.result["_2"].max()
        self.assertEqual(res1, res2)


if __name__ == "__main__":
    ProgressiveTest.main()
