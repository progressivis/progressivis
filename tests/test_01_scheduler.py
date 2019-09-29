from . import ProgressiveTest
from time import sleep

from progressivis import Print, Scheduler
from progressivis.io import CSVLoader
from progressivis.stats import Min
from progressivis.datasets import get_dataset


class TestScheduler(ProgressiveTest):

    def test_scheduler(self):
        s = Scheduler()
        csv = CSVLoader(get_dataset('bigfile'),
                        name="csv",
                        index_col=False, header=None,
                        scheduler=s)
        self.assertIs(s["csv"], csv)
        s.start()
        print('back from start')

        # sleep(1)
        self.assertTrue(csv.scheduler().is_running())

        def add_min():
            with s:
                m = Min(scheduler=s)
                m.input.table = csv.output.table
                prt = Print(proc=self.terse, scheduler=s)
                prt.input.df = m.output.table

        s.on_tick_once(add_min)

        sleep(1)
        s.stop()
        s.join()


if __name__ == '__main__':
    ProgressiveTest.main()
