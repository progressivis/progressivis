from progressivis.table.constant import Constant
from progressivis.table.table import Table
from progressivis.table.paste import Paste
from progressivis import Print
from progressivis.stats import  RandomTable, Min, Max
from progressivis.core.bitmap import bitmap
from progressivis.table.range_query import RangeQuery
import numpy as np
from . import ProgressiveTest, main
class TestPaste(ProgressiveTest):    
    def test_paste(self):
        s = self.scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        min_1 = Min(name='min_1'+str(hash(random)), scheduler=s, columns=['_1'])
        min_1.input.table = random.output.table
        min_2 = Min(name='min_2'+str(hash(random)), scheduler=s, columns=['_2'])
        min_2.input.table = random.output.table
        bj = Paste(scheduler=s)
        bj.input.first = min_1.output.table
        bj.input.second = min_2.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = bj.output.table
        s.start()
        s.join()
        res1 = random.table().min()
        res2 = bj.table().last().to_dict()
        self.assertAlmostEqual(res1['_1'], res2['_1'])
        self.assertAlmostEqual(res1['_2'], res2['_2'])
        
