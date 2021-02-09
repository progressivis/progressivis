from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis.table.dict2table import Dict2Table
from progressivis.table.paste import Paste
from progressivis import Print
from progressivis.stats import  RandomTable, Min, Max
from progressivis.core.bitmap import bitmap
from progressivis.table.range_query import RangeQuery
from progressivis.core import aio
import numpy as np


from . import ProgressiveTest, main
class TestPaste(ProgressiveTest):
    def test_paste(self):
        s = self.scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        min_1 = Min(name='min_1'+str(hash(random)), scheduler=s, columns=['_1'])
        min_1.input.table = random.output.result
        d2t_1 = Dict2Table(scheduler=s)
        d2t_1.input.dict_ = min_1.output.result
        min_2 = Min(name='min_2'+str(hash(random)), scheduler=s, columns=['_2'])
        min_2.input.table = random.output.result
        d2t_2 = Dict2Table(scheduler=s)
        d2t_2.input.dict_ = min_2.output.result
        bj = Paste(scheduler=s)
        bj.input.first = d2t_1.output.result
        bj.input.second = d2t_2.output.result
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = bj.output.result
        aio.run(s.start())
        res1 = random.result.min()
        res2 = bj.result.last().to_dict()
        self.assertAlmostEqual(res1['_1'], res2['_1'])
        self.assertAlmostEqual(res1['_2'], res2['_2'])

