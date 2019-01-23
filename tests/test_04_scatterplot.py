from progressivis import Every, Print
from progressivis.io import CSVLoader
from progressivis.vis import MCScatterPlot
from progressivis.datasets import get_dataset
from progressivis.stats import  RandomTable
from progressivis.core.utils import decorate, ModulePatch

from . import ProgressiveTest, skip

#from pprint import pprint

def print_len(x):
    if x is not None:
        print(len(x))

def print_repr(x):
    if x is not None:
        print(repr(x))

def idle_proc(s, _):
    s.stop()


LOWER_X = 0.2
LOWER_Y = 0.3
UPPER_X = 0.8
UPPER_Y = 0.7

class VariablePatch1(ModulePatch):
    def before_run_step(self, m, *args, **kwargs):
        if m._table:
            m.from_input({'_1': LOWER_X, '_2': LOWER_Y})

class VariablePatch2(ModulePatch):
    def before_run_step(self, m, *args, **kwargs):
        if m._table:
            m.from_input({'_1': UPPER_X, '_2': UPPER_Y})

class ScatterPlotPatch(ModulePatch):
    def __init__(self, n):
        super(ScatterPlotPatch, self).__init__(n)
        self._last_run = 0

    def after_run_step(self, m, *args, **kwargs):
        scheduler = m.scheduler()
        # Deciding when to stop is tricky for now
        if self._last_run+4 == scheduler.run_number():
            m.scheduler().stop()
        else:
            self._last_run = scheduler.run_number()

class TestScatterPlot(ProgressiveTest):
#    def setUp(self):
#        log_level(logging.INFO,'progressivis')
    def tearDown(self):
        TestScatterPlot.cleanup()
    def test_scatterplot(self):
        s = self.scheduler()
        csv = CSVLoader(get_dataset('smallfile'),
                        index_col=False, header=None,
                        force_valid_ids=True, scheduler=s)
        sp = MCScatterPlot(scheduler=s, classes=[('Scatterplot', '_1', '_2')], approximate=True)
        sp.create_dependent_modules(csv, 'table')
        cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
        cnt.input.df = csv.output.table
        prt = Print(proc=self.terse, scheduler=s)
        prt.input.df = sp.output.table
        csv.scheduler().start(idle_proc=idle_proc)
        s.join()
        self.assertEqual(len(csv.table()), 30000)

    def test_scatterplot2(self):
        s = self.scheduler()
        random = RandomTable(2, rows=2000000, scheduler=s)
        sp = MCScatterPlot(scheduler=s, classes=[('Scatterplot', '_1', '_2')], approximate=True)
        sp.create_dependent_modules(random, 'table', with_sampling=False)
        cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
        cnt.input.df = random.output.table
        prt = Print(proc=self.terse, scheduler=s)
        prt.input.df = sp.output.table
        decorate(s, VariablePatch1("variable_1"))
        decorate(s, VariablePatch2("variable_2"))
        decorate(s, ScatterPlotPatch("mc_scatter_plot_1"))
        sp.scheduler().start(idle_proc=idle_proc)
        s.join()
        js = sp.to_json()
        x, y, _ = zip(*js['sample']['data'])
        min_x=min(x)
        max_x=max(x)
        min_y=min(y)
        max_y=max(y)
        self.assertGreaterEqual(min_x, LOWER_X)
        self.assertGreaterEqual(min_y, LOWER_Y)
        self.assertLessEqual(max_x, UPPER_X)
        self.assertLessEqual(max_y, UPPER_Y)

if __name__ == '__main__':
    ProgressiveTest.main()

