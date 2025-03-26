from . import ProgressiveTest, skipIf

from progressivis import Print, CSVLoader, Sample
from progressivis.datasets import get_dataset
from progressivis.core import aio
from progressivis.stats.cxxsample import CxxSample  # type: ignore

from typing import Any


def print_repr(x: Any) -> None:
    print(repr(x))


class TestCxxSample(ProgressiveTest):
    @skipIf(CxxSample is None, "C++ module is missing")
    def test_sample(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        smp = Sample(samples=10, scheduler=s)
        smp.input[0] = csv.output.result
        prt = Print(proc=self.terse, scheduler=s)
        prt.input[0] = smp.output.result
        aio.run(csv.scheduler().start())
        assert smp.result is not None
        self.assertEqual(len(smp.result), 10)


if __name__ == "__main__":
    ProgressiveTest.main()
