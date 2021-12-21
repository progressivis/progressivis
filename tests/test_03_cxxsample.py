from . import ProgressiveTest, skipIf

from progressivis import Print
from progressivis.io import CSVLoader
from progressivis.stats import Sample
from progressivis.datasets import get_dataset
from progressivis.core import aio
from progressivis.stats.cxxsample import CxxSample


def print_repr(x):
    print(repr(x))


class TestCxxSample(ProgressiveTest):
    #    def setUp(self):
    #        log_level(logging.INFO)
    @skipIf(CxxSample is None, "C++ module is missing")
    def test_sample(self):
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"), index_col=False, header=None, scheduler=s
        )
        smp = Sample(samples=10, scheduler=s)
        smp.input[0] = csv.output.result
        prt = Print(proc=self.terse, scheduler=s)
        prt.input[0] = smp.output.result
        aio.run(csv.scheduler().start())
        # print(repr(smp.result))
        self.assertEqual(len(smp.result), 10)


if __name__ == "__main__":
    ProgressiveTest.main()
