from time import sleep

from progressivis import Print, Scheduler
from progressivis.io import CSVLoader
from progressivis.stats import Min
from progressivis.datasets import get_dataset
from progressivis.core.dataflow import Dataflow

from . import ProgressiveTest, skip


class TestDataflow(ProgressiveTest):
    @skip("Not ready yet")
    def test_dataflow(self):
        s = Scheduler()
        with Dataflow(s):
            csv = CSVLoader(get_dataset('bigfile'), name="csv", index_col=False, header=None)
            m = Min()
            m.input.table = csv.output.table
            prt = Print(proc=self.terse)
            prt.input.df = m.output.table

        self.assertIs(s["csv"], csv)
        csv.scheduler().start()

        sleep(1)
        self.assertTrue(csv.scheduler().is_running())

        s.stop()
        s.join()

if __name__ == '__main__':
    ProgressiveTest.main()
