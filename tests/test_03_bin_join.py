from progressivis.table.bin_join import BinJoin
from progressivis import Print
from progressivis.stats import RandomTable, Min
from . import ProgressiveTest


class TestBinJoin(ProgressiveTest):
    def test_bin_join(self):
        s = self.scheduler()
        random = RandomTable(10, rows=10000, scheduler=s)
        min_1 = Min(name='min_1'+str(hash(random)), columns=['_1'],
                    scheduler=s)
        min_1.input.table = random.output.table
        min_2 = Min(name='min_2'+str(hash(random)), columns=['_2'],
                    scheduler=s)
        min_2.input.table = random.output.table
        bj = BinJoin(scheduler=s)
        bj.input.first = min_1.output.table
        bj.input.second = min_2.output.table
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = bj.output.table
        s.start()
        s.join()
        res1 = random.table().min()
        res2 = bj.table().last().to_dict()
        self.assertAlmostEqual(res1['_1'], res2['_1'])
        self.assertAlmostEqual(res1['_2'], res2['_2'])
