from progressivis import Every, Print
from progressivis.io import CSVLoader
from progressivis.vis import MCScatterPlot
from progressivis.datasets import get_dataset
from progressivis.stats import RandomTable
from progressivis.core import aio
from . import ProgressiveTest, skipIf
import os

def print_len(x):
    if x is not None:
        print(len(x))


def print_repr(x):
    if x is not None:
        print(repr(x))

#async def idle_proc(s, _):
#    await s.stop()


LOWER_X = 0.2
LOWER_Y = 0.3
UPPER_X = 0.8
UPPER_Y = 0.7


async def fake_input(sched, name, t, inp):
    await aio.sleep(t)
    module = sched.modules()[name]
    await module.from_input(inp)


async def sleep_then_stop(s, t):
    await aio.sleep(t)
    await s.stop()
    print(s._run_list)


class TestScatterPlot(ProgressiveTest):
    def tearDown(self):
        TestScatterPlot.cleanup()

    def test_scatterplot(self):
        s = self.scheduler(clean=True)
        with s:
            csv = CSVLoader(get_dataset('smallfile'),
                            index_col=False, header=None,
                            force_valid_ids=True, scheduler=s)
            sp = MCScatterPlot(scheduler=s,
                               classes=[('Scatterplot', '_1', '_2')],
                               approximate=True)
            sp.create_dependent_modules(csv, 'result')
            cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
            cnt.input[0] = csv.output.result
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = sp.output.result
            sts = sleep_then_stop(s, 5)
        aio.run_gather(csv.scheduler().start(), sts)
        self.assertEqual(len(csv.result), 30000)

    @skipIf(os.getenv('TRAVIS'), 'skipped because is killed (sometimes) by the system on CI')
    def test_scatterplot2(self):
        s = self.scheduler(clean=True)
        with s:
            random = RandomTable(2, rows=2000000, scheduler=s)
            sp = MCScatterPlot(scheduler=s,
                               classes=[('Scatterplot', '_1', '_2')],
                               approximate=True)
            sp.create_dependent_modules(random, 'result', with_sampling=False)
            cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
            cnt.input[0] = random.output.result
            prt = Print(proc=self.terse, scheduler=s)
            prt.input[0] = sp.output.result
        finp1 = fake_input(s, "variable_1", 6, {'_1': LOWER_X, '_2': LOWER_Y})
        finp2 = fake_input(s, "variable_2", 6, {'_1': UPPER_X, '_2': UPPER_Y})
        sts = sleep_then_stop(s, 20)
        aio.run_gather(sp.scheduler().start(), finp1, finp2, sts)
        js = sp.to_json()
        x, y, _ = zip(*js['sample']['data'])
        min_x = min(x)
        max_x = max(x)
        min_y = min(y)
        max_y = max(y)
        self.assertGreaterEqual(min_x, LOWER_X)
        self.assertGreaterEqual(min_y, LOWER_Y)
        self.assertLessEqual(max_x, UPPER_X)
        self.assertLessEqual(max_y, UPPER_Y)


if __name__ == '__main__':
    ProgressiveTest.main()
