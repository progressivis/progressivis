from . import ProgressiveTest
from progressivis.core import aio
from progressivis import Every
from progressivis.stats.random_table import RandomPTable

from typing import Any


def print_len(x: Any) -> None:
    if x is not None:
        print(len(x))


class TestRandomPTable(ProgressiveTest):
    def test_random_table(self) -> None:
        s = self.scheduler()
        module = RandomPTable(["a", "b"], rows=10000, scheduler=s)
        assert module.result is not None
        self.assertEqual(module.result.columns[0], "a")
        self.assertEqual(module.result.columns[1], "b")
        self.assertEqual(len(module.result.columns), 2)  # add the UPDATE_COLUMN
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = module.output.result
        aio.run(s.start())

        self.assertEqual(len(module.result), 10000)

    def test_random_table2(self) -> None:
        s = self.scheduler()
        # produces more than 4M rows per second on my laptop
        module = RandomPTable(10, rows=1000000, force_valid_ids=True, scheduler=s)
        assert module.result is not None
        self.assertEqual(len(module.result.columns), 10)  # add the UPDATE_COLUMN
        self.assertEqual(module.result.columns[0], "_1")
        self.assertEqual(module.result.columns[1], "_2")
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = module.output.result
        aio.run(s.start())
        assert module.result is not None
        self.assertEqual(len(module.result), 1000000)
