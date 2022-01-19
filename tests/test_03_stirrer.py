from . import ProgressiveTest
from progressivis import Print, Scheduler
from progressivis.stats import RandomTable, Max
from progressivis.table.stirrer import Stirrer
from progressivis.core import aio

from typing import Any

class TestStirrer(ProgressiveTest):
    def test_stirrer(self) -> None:
        s = Scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(
            update_column="_1",
            delete_rows=5,
            update_rows=5,
            fixed_step_size=100,
            scheduler=s,
        )
        stirrer.input[0] = random.output.result
        max_ = Max(name="max_" + str(hash(random)), scheduler=s)
        max_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = max_.output.result
        aio.run(s.start())
        res1 = stirrer.result.max()
        res2 = max_.result
        self.compare(res1, res2)

    def compare(self, res1: Any, res2: Any) -> None:
        v1 = np.array(list(res1.values()))
        v2 = np.array(list(res2.values()))
        self.assertEqual(v1.shape, v2.shape)
        #print('v1 = ', v1)
        #print('v2 = ', v2)
        self.assertTrue(np.allclose(v1, v2))



if __name__ == "__main__":
    ProgressiveTest.main()
