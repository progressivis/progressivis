from progressivis.table.table import Table
from progressivis.table.constant import Constant
from progressivis import Print
from progressivis.stats import RandomTable
from progressivis.table.bisectmod import Bisect
from progressivis.core.bitmap import bitmap
from progressivis.table.hist_index import HistogramIndex
from progressivis.table.intersection import Intersection
import asyncio as aio

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
        bisect_min.input.table = hist_index.output.table
        # bisect_.input.table = random.output.table
        bisect_min.input.limit = min_value.output.table
        bisect_max = Bisect(column='_1', op='<', hist_index=hist_index,
                            scheduler=s)
        bisect_max.input.table = hist_index.output.table
        # bisect_.input.table = random.output.table
        bisect_max.input.limit = max_value.output.table
        inter = Intersection(scheduler=s)
        inter.input.table = bisect_min.output.table
        inter.input.table = bisect_max.output.table
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = inter.output.table
        aio.run(s.start())
        idx = hist_index.input_module\
                        .output['table']\
                        .data().eval('(_1>0.3)&(_1<0.8)',
                                     result_object='index')
        self.assertEqual(inter.table().selection, bitmap(idx))


if __name__ == '__main__':
    ProgressiveTest.main()
