"Test for Range Query"
#import numpy as np
from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis import Print
from progressivis.stats import  RandomTable
#from progressivis.table.bisectmod import Bisect
from progressivis.core.bitmap import bitmap
from progressivis.table.range_query import RangeQuery
from . import ProgressiveTest, main
from stool.bench_it import decorate

class TestRangeQuery(ProgressiveTest):
    "Test Suite for RangeQuery Module"
    def test_range_query(self):
        "Run tests of the RangeQuery module"
        s = self.scheduler
        random = RandomTable(2, rows=100000, scheduler=s)
        t_min = Table(name=None, dshape='{_1: float64}', data={'_1':[0.3]})
        min_value = Constant(table=t_min, scheduler=s)
        t_max = Table(name=None, dshape='{_1: float64}', data={'_1':[0.8]})
        max_value = Constant(table=t_max, scheduler=s)
        range_qry = RangeQuery(column='_1', scheduler=s)
        range_qry.create_dependent_modules(random, 'table',
                                           min_value=min_value,
                                           max_value=max_value)
        prt = Print(proc=self.terse, scheduler=s)
        prt.input.df = range_qry.output.table
        env = decorate(s, '/tmp/toto.db', drop_existing_db=True)
        s.start()
        s.join()
        #dump_table('measurement_tbl', '/tmp/toto.db')
        env.dump()
        idx = range_qry.input_module.output['table']\
          .data().eval('(_1>0.3)&(_1<0.8)', result_object='index')
        self.assertEqual(range_qry.table().selection, bitmap(idx))


if __name__ == '__main__':
    main()
