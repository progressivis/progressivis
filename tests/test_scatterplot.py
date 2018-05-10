from progressivis import Every, Print
from progressivis.io import CSVLoader
from progressivis.vis import ScatterPlot
from progressivis.datasets import get_dataset
from progressivis.stats import  RandomTable
from progressivis.core.utils import decorate, ModulePatch

from . import ProgressiveTest

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

class SentinelPatch(ModulePatch):
    def __init__(self, n):
        SentinelPatch.cnt = 0
        super(SentinelPatch, self).__init__(n)
    def after_run_step(self, m, *args, **kwargs):
        if SentinelPatch.cnt > 10:
            m.scheduler().stop()
        else:
            SentinelPatch.cnt += 1

class TestScatterPlot(ProgressiveTest):
#    def setUp(self):
#        log_level(logging.INFO,'progressivis')

    def test_scatterplot(self):
        s = self.scheduler()
        csv = CSVLoader(get_dataset('smallfile'),
                        index_col=False, header=None,
                        force_valid_ids=True, scheduler=s)
        sp = ScatterPlot(x_column='_1', y_column='_2', scheduler=s)
        sp.create_dependent_modules(csv, 'table')
        cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
        cnt.input.df = csv.output.table
        prt = Print(proc=self.terse, scheduler=s)
        prt.input.df = sp.output.table
        csv.scheduler().start(idle_proc=idle_proc)
        s.join()
        self.assertEqual(len(csv.table()), 30000) #1000000)
        #pprint(sp.to_json())

    def test_scatterplot2(self):
        s = self.scheduler()
        random = RandomTable(2, rows=2000000, scheduler=s)
        sp = ScatterPlot(x_column='_1', y_column='_2', scheduler=s)
        sp.create_dependent_modules(random, 'table', with_sampling=False)
        cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
        cnt.input.df = random.output.table
        prt = Print(proc=self.terse, scheduler=s)
        prt.input.df = sp.output.table
        decorate(s, VariablePatch1("variable_1"))
        decorate(s, VariablePatch2("variable_2"))
        decorate(s, SentinelPatch("sentinel_1"))
        sp.scheduler().start(idle_proc=idle_proc)
        s.join()
        #import pdb; pdb.set_trace()
        js = sp.to_json()
        x, y = zip(*js['scatterplot']['data'])
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

