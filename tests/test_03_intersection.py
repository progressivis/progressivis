from progressivis.table.table import Table
from progressivis.table.constant import Constant
from progressivis import Print
from progressivis.stats import RandomTable
from progressivis.table.bisectmod import Bisect
from progressivis.core.bitmap import bitmap
from progressivis.table.hist_index import HistogramIndex
from progressivis.table.intersection import Intersection
from progressivis.table.stirrer import Stirrer
from progressivis.core import aio

from . import ProgressiveTest


class TestIntersection(ProgressiveTest):
    def test_intersection(self):
        s = self.scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        t_min = Table(name=None,
                      dshape='{_1: float64}', data={'_1': [0.3]})
        min_value = Constant(table=t_min, scheduler=s)
        t_max = Table(name=None,
                      dshape='{_1: float64}', data={'_1': [0.8]})
        max_value = Constant(table=t_max, scheduler=s)
        hist_index = HistogramIndex(column='_1', scheduler=s)
        hist_index.create_dependent_modules(random, 'table')
        bisect_min = Bisect(column='_1', op='>', hist_index=hist_index,
                            scheduler=s)
        bisect_min.input.table = hist_index.output.result
        # bisect_.input.table = random.output.result
        bisect_min.input.limit = min_value.output.result
        bisect_max = Bisect(column='_1', op='<', hist_index=hist_index,
                            scheduler=s)
        bisect_max.input.table = hist_index.output.result
        # bisect_.input.table = random.output.result
        bisect_max.input.limit = max_value.output.result
        inter = Intersection(scheduler=s)
        inter.input.table = bisect_min.output.result
        inter.input.table = bisect_max.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = inter.output.result
        aio.run(s.start())
        idx = hist_index.input_module\
                        .output['table']\
                        .data().eval('(_1>0.3)&(_1<0.8)',
                                     result_object='index')
        self.assertEqual(inter.result.index, bitmap(idx))

    def _impl_stirred_tst_intersection(self, **kw):
        s = self.scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(update_column='_2',
                          fixed_step_size=1000, scheduler=s, **kw)
        stirrer.input.table = random.output.result
        t_min = Table(name=None,
                      dshape='{_1: float64}', data={'_1': [0.3]})
        min_value = Constant(table=t_min, scheduler=s)
        t_max = Table(name=None,
                      dshape='{_1: float64}', data={'_1': [0.8]})
        max_value = Constant(table=t_max, scheduler=s)
        hist_index = HistogramIndex(column='_1', scheduler=s)
        hist_index.create_dependent_modules(stirrer, 'table')
        bisect_min = Bisect(column='_1', op='>', hist_index=hist_index,
                            scheduler=s)
        bisect_min.input.table = hist_index.output.result
        # bisect_.input.table = random.output.result
        bisect_min.input.limit = min_value.output.result
        bisect_max = Bisect(column='_1', op='<', hist_index=hist_index,
                            scheduler=s)
        bisect_max.input.table = hist_index.output.result
        # bisect_.input.table = random.output.result
        bisect_max.input.limit = max_value.output.result
        inter = Intersection(scheduler=s)
        inter.input.table = bisect_min.output.result
        inter.input.table = bisect_max.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = inter.output.result
        aio.run(s.start())
        idx = hist_index.input_module\
                        .output['table']\
                        .data().eval('(_1>0.3)&(_1<0.8)',
                                     result_object='index')
        self.assertEqual(inter.result.index, bitmap(idx))

    def test_intersection2(self):
        self._impl_stirred_tst_intersection(delete_rows=5)

    def test_intersection3(self):
        self._impl_stirred_tst_intersection(update_rows=5)

if __name__ == '__main__':
    ProgressiveTest.main()
