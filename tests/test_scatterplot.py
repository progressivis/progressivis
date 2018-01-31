from . import ProgressiveTest

from progressivis import Every, Print
from progressivis.io import CSVLoader
from progressivis.vis import ScatterPlot
from progressivis.datasets import get_dataset
#from pprint import pprint

def print_len(x):
    if x is not None:
        print(len(x))

def print_repr(x):
    if x is not None:
        print(repr(x))
    
def idle_proc(s, _):
    s.stop()

class TestScatterPlot(ProgressiveTest):
#    def setUp(self):
#        log_level(logging.INFO,'progressivis')

    def test_scatterplot(self):
        s = self.scheduler()
        csv = CSVLoader(get_dataset('smallfile'),index_col=False,header=None,force_valid_ids=True,scheduler=s)
        sp = ScatterPlot(x_column='_1', y_column='_2', scheduler=s)
        sp.create_dependent_modules(csv,'table')
        cnt = Every(proc=self.terse, constant_time=True,scheduler=s)
        cnt.input.df = csv.output.table
        prt = Print(proc=self.terse, scheduler=s)
        prt.input.df = sp.output.table
        csv.scheduler().start(idle_proc=idle_proc)
        s.join()
        self.assertEqual(len(csv.table()), 30000) #1000000)
        #pprint(sp.to_json())


if __name__ == '__main__':
    ProgressiveTest.main()
