from __future__ import annotations

from . import ProgressiveTest

from progressivis import Every, LastRow, CSVLoader, get_dataset
from progressivis.core.api import notNone
from progressivis.core import aio


class TestLastRow(ProgressiveTest):
    def test_last_row(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("smallfile"), header=None, scheduler=s
        )
        lr1 = LastRow(scheduler=s)
        lr1.input[0] = csv.output.result
        prlen = Every(proc=self.terse, constant_time=True, scheduler=s)
        prlen.input[0] = lr1.output.result
        aio.run(s.start())
        assert csv.result is not None
        df = csv.result
        res = lr1.result
        assert res is not None
        self.assertEqual(res.at[0, "_1"], notNone(df.last())["_1"])


if __name__ == "__main__":
    ProgressiveTest.main()
