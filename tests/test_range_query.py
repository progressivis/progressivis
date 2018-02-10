from . import ProgressiveTest
from progressivis.table.table import Table
from progressivis.io import Variable
from progressivis import Print, Scheduler
from progressivis.stats import  RandomTable
#from progressivis.table.bisectmod import Bisect
from progressivis.core.bitmap import bitmap
from progressivis.table.range_query import RangeQuery

import numpy as np

class TestRangeQuery(ProgressiveTest):
    def test_range_query(self):
        s=self.scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        t_min = Table(name=None, dshape='{value: string}', data={'value':[0.3]})
        min_value = Variable(table=t_min, scheduler=s)
        t_max = Table(name=None, dshape='{value: string}', data={'value':[0.8]})
        max_value = Variable(table=t_max, scheduler=s)
        range_qry = RangeQuery(column='_1', scheduler=s)
        range_qry.create_dependent_modules(random, 'table', min_value=min_value, max_value=max_value)
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = range_qry.output.table
        s.start()
        s.join()
        idx = range_qry.input_module.output['table'].data().eval('(_1>0.3)&(_1<0.8)', result_object='index')
        self.assertEqual(range_qry._table.selection, bitmap(idx))


if __name__ == '__main__':
    unittest.main()
