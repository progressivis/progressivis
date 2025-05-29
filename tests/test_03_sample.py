from __future__ import annotations

from . import ProgressiveTest

from progressivis import Print, CSVLoader, Sample, get_dataset
from progressivis.core import aio

from typing import Any


def print_repr(x: Any) -> None:
    print(repr(x))


class TestSample(ProgressiveTest):
    def test_sample(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("smallfile"), header=None, scheduler=s
        )
        smp = Sample(samples=10, scheduler=s)
        smp.input[0] = csv.output.result
        prt = Print(proc=self.terse, scheduler=s)
        prt.input[0] = smp.output.result
        prt2 = Print(proc=self.terse, scheduler=s)
        prt2.input[0] = smp.output.select
        aio.run(csv.scheduler().start())
        assert smp.result is not None
        self.assertEqual(len(smp.result), 10)
        assert smp.pintset is not None
        self.assertEqual(len(smp.pintset), 10)


if __name__ == "__main__":
    ProgressiveTest.main()
