from . import ProgressiveTest

from time import sleep

from progressivis import Print, Scheduler, ProgressiveError
from progressivis.io import CSVLoader
from progressivis.stats import Min, RandomTable
from progressivis.datasets import get_dataset
from progressivis.core import aio, SlotDescriptor, Module


class TestModule(Module):
    inputs = [
        SlotDescriptor('a'),
        SlotDescriptor('b', required=False)
    ]
    outputs = [
        SlotDescriptor('c'),
        SlotDescriptor('d', required=False)
    ]

    def __init__(self, **kwds):
        super(TestModule, self).__init__(**kwds)

    def run_step(self, run_number, step_size, howlong):  # pragma no cover
        return self._return_run_step(self.state_blocked, 0)


class TestScheduler(ProgressiveTest):
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

        self.assertTrue(check_running)

        def add_min():
            with s:
                m = Min(scheduler=s)
                m.input.table = csv.output.result
                prt = Print(proc=self.terse, scheduler=s)
                prt.input.df = m.output.result

        s.on_tick_once(add_min)

        sleep(1)
        s.task_stop()

        self.assertIs(s['csv'], csv)
        json = s.to_json(short=False)
        self.assertFalse(json['is_running'])
        self.assertTrue(json['is_terminated'])
        html = s._repr_html_()
        self.assertTrue(len(html) != 0)

    def test_scheduler_dels(self):
        s = Scheduler()
        table = RandomTable(name='table', columns=['a'], scheduler=s)
        m = Min(name='min', scheduler=s)
        m.input.table = table.output.result
        prt = Print(name='prt', scheduler=s)
        prt.input.df = m.output.result

        aio.run(s.step())
        deps = s.collateral_damage('table')
        self.assertEquals(deps, set(['table', 'min', 'prt']))

    def test_scheduler_dels2(self):
        s = Scheduler()
        table = RandomTable(name='table', columns=['a'], scheduler=s)
        m = TestModule(name='min', scheduler=s)
        m.input.a = table.output.result
        prt = Print(name='prt', scheduler=s)
        prt.input.df = m.output.c
        # from nose.tools import set_trace; set_trace()
        s.commit()
        aio.run(s.step())
        deps = s.collateral_damage('table')
        self.assertEquals(deps, set(['table', 'min', 'prt']))


if __name__ == '__main__':
    ProgressiveTest.main()
