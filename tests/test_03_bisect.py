from progressivis.table.table import Table
from progressivis.table.constant import Constant
from progressivis import Print
from progressivis.stats import  RandomTable
from progressivis.table.bisectmod import Bisect
from progressivis.core.bitmap import bitmap
from progressivis.table.hist_index import HistogramIndex
from progressivis.core import aio
from progressivis.table.stirrer import Stirrer
from . import ProgressiveTest



class TestBisect(ProgressiveTest):
    def test_bisect(self):
        s = self.scheduler()
        random = RandomTable(2, rows=1000_000, scheduler=s)
        t = Table(name=None, dshape='{value: string}', data={'value':[0.5]})
        min_value = Constant(table=t, scheduler=s) 
        hist_index = HistogramIndex(column='_1', scheduler=s)
        hist_index.create_dependent_modules(random, 'table')
        bisect_ = Bisect(column='_1', op='>', hist_index=hist_index, scheduler=s)
        bisect_.input.table = hist_index.output.table
        #bisect_.input.table = random.output.table
        bisect_.input.limit = min_value.output.table
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = bisect_.output.table
        aio.run(s.start())
        #hist_index._impl.dump()
        idx = random.result.eval('_1>0.5', result_object='index')
        self.assertEqual(bisect_.result.index, bitmap(idx))

    def test_bisect2(self):
        s = self.scheduler()
        random = RandomTable(2, rows=100_000, scheduler=s)
        stirrer = Stirrer(update_column='_1', delete_rows=100,
                          #update_rows=5,
                          #fixed_step_size=100,
                          scheduler=s)
        stirrer.input.table = random.output.table        
        t = Table(name=None, dshape='{value: string}', data={'value':[0.5]})
        min_value = Constant(table=t, scheduler=s) 
        hist_index = HistogramIndex(column='_1', scheduler=s)
        hist_index.create_dependent_modules(stirrer, 'table')
        bisect_ = Bisect(column='_1', op='>', hist_index=hist_index, scheduler=s)
        bisect_.input.table = hist_index.output.table
        #bisect_.input.table = random.output.table
        bisect_.input.limit = min_value.output.table
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = bisect_.output.table
        aio.run(s.start())
        idx = stirrer.result.eval('_1>0.5', result_object='index')
        self.assertEqual(bisect_.result.index, bitmap(idx))

    def test_bisect3(self):
        s = self.scheduler()
        random = RandomTable(2, rows=100_000, scheduler=s)
        stirrer = Stirrer(update_column='_1', update_rows=100,
                          fixed_step_size=100, scheduler=s)
        stirrer.input.table = random.output.table
        t = Table(name=None, dshape='{value: string}', data={'value':[0.5]})
        min_value = Constant(table=t, scheduler=s)
        hist_index = HistogramIndex(column='_1', scheduler=s)
        hist_index.create_dependent_modules(stirrer, 'table')
        bisect_ = Bisect(column='_1', op='>', hist_index=hist_index, scheduler=s)
        bisect_.input.table = hist_index.output.table
        #bisect_.input.table = random.output.table
        bisect_.input.limit = min_value.output.table
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = bisect_.output.table
        aio.run(s.start())
        idx = stirrer.result.eval('_1>0.5', result_object='index')
        self.assertEqual(bisect_.result.index, bitmap(idx))


if __name__ == '__main__':
    ProgressiveTest.main()
