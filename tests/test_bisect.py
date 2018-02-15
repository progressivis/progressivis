from progressivis.table.table import Table
from progressivis.table.constant import Constant
from progressivis import Print
from progressivis.stats import  RandomTable
from progressivis.table.bisectmod import Bisect
from progressivis.core.bitmap import bitmap
from progressivis.table.hist_index import HistogramIndex

from . import ProgressiveTest



class TestBisect(ProgressiveTest):
    def test_bisect(self):
        s = self.scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
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
        s.start()
        s.join()
        idx = bisect_.get_input_slot('table').data().eval('_1>0.5', result_object='index')
        self.assertEqual(bisect_._table.selection, bitmap(idx))


if __name__ == '__main__':
    ProgressiveTest.main()
