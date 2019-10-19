from . import ProgressiveTest
import asyncio as aio
from progressivis import Print
from progressivis.table import Table
from progressivis.table.cmp_query import CmpQueryLast
from progressivis.table.constant import Constant
from progressivis.stats import RandomTable



class TestCmpQuery(ProgressiveTest):
    def test_cmp_query(self):
        s=self.scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        cmp_ = CmpQueryLast(scheduler=s)
        cst = Table("cmp_table", data={'_1': [0.5]})
        value = Constant(cst, scheduler=s)
        cmp_.input.cmp = value.output.table
        cmp_.input.table = random.output.table
        pr=Print(proc=self.terse, scheduler=s)
        pr.input.df = cmp_.output.select
        aio.run(s.start())
        #s.join()


if __name__ == '__main__':
    ProgressiveTest.main()
