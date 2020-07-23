from . import ProgressiveTest, skip

from time import sleep

from progressivis import Print, Scheduler, ProgressiveError
from progressivis.io import CSVLoader
from progressivis.stats import Min
from progressivis.datasets import get_dataset
from progressivis.core import aio

class TestScheduler(ProgressiveTest):
    #@skip("Needs fixing")
    def test_scheduler(self):
        with self.assertRaises(ProgressiveError):
            s = Scheduler(0)
        s = Scheduler()
        csv = CSVLoader(get_dataset('bigfile'),
                        name="csv",
                        index_col=False, header=None,
                        scheduler=s)
        self.assertIs(s["csv"], csv)
        check_running = False
        async def _is_running():
            nonlocal check_running
            check_running = csv.scheduler().is_running()
        aio.run_gather(s.start(), _is_running())
        #sleep(1)
        self.assertTrue(check_running)
        def add_min():
            with s:
                m = Min(scheduler=s)
                m.input.table = csv.output.table
                prt = Print(proc=self.terse, scheduler=s)
                prt.input.df = m.output.table

        s.on_tick_once(add_min)

        sleep(1)
        s.task_stop()
        #s.join()
        self.assertIs(s['csv'], csv)
        json = s.to_json(short=False)
        self.assertFalse(json['is_running'])
        self.assertTrue(json['is_terminated'])
        html = s._repr_html_()
        self.assertTrue(len(html) != 0)


if __name__ == '__main__':
    ProgressiveTest.main()
