from progressivis.table.bin_join import BinJoin
from progressivis import Print
from progressivis.stats import RandomPTable, Min
from progressivis.table.dict2table import Dict2PTable
from progressivis.core import aio, notNone

from . import ProgressiveTest


class TestBinJoin(ProgressiveTest):
    def test_bin_join(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10000, scheduler=s)
        min_1 = Min(name="min_1" + str(hash(random)), columns=["_1"], scheduler=s)
        min_1.input[0] = random.output[0]
        d2t_1 = Dict2PTable(scheduler=s)
        d2t_1.input.dict_ = min_1.output[0]
        min_2 = Min(name="min_2" + str(hash(random)), columns=["_2"], scheduler=s)
        min_2.input[0] = random.output[0]
        d2t_2 = Dict2PTable(scheduler=s)
        d2t_2.input.dict_ = min_2.output[0]
        bj = BinJoin(scheduler=s)
        bj.input.first = d2t_1.output[0]
        bj.input.second = d2t_2.output[0]
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = bj.output[0]
        aio.run(s.start())
        assert random.result is not None
        assert bj.result is not None
        res1 = random.result.min()
        res2 = notNone(bj.result.last()).to_dict()
        self.assertAlmostEqual(res1["_1"], res2["_1"])
        self.assertAlmostEqual(res1["_2"], res2["_2"])
