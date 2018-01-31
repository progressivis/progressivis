from . import ProgressiveTest

from progressivis import Print
from progressivis.io import CSVLoader
from progressivis.stats import Sample
from progressivis.datasets import get_dataset


def print_repr(x):
    print(repr(x))


class TestSample(ProgressiveTest):
#    def setUp(self):
#        log_level(logging.INFO)

    def test_sample(self):
        s = self.scheduler()
        csv = CSVLoader(get_dataset('bigfile'), index_col=False, header=None, scheduler=s)
        smp = Sample(samples=10,scheduler=s)
        smp.input.table = csv.output.table
        prt = Print(proc=self.terse, scheduler=s)
        prt.input.df = smp.output.table
        csv.scheduler().start()
        s.join()
        #print(repr(smp.table()))
        self.assertEqual(len(smp.table()), 10)

if __name__ == '__main__':
    ProgressiveTest.main()
