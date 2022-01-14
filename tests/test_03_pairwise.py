from __future__ import annotations

from . import ProgressiveTest

from progressivis import Every
from progressivis.io import VECLoader, CSVLoader

# from progressivis.metrics import PairwiseDistances
from progressivis.datasets import get_dataset
from progressivis.core import aio

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis import Scheduler


def print_len(x: Any) -> None:
    if x is not None:
        print(len(x))


times = 0


async def ten_times(scheduler: Scheduler, run_number: int) -> None:
    global times
    times += 1
    # import pdb;pdb.set_trace()
    if times > 10:
        await scheduler.stop()


class TestPairwiseDistances(ProgressiveTest):
    def NOtest_vec_distances(self) -> None:
        s = self.scheduler()
        vec = VECLoader(get_dataset("warlogs"), scheduler=s)
        #        dis=PairwiseDistances(metric='cosine',scheduler=s)
        #        dis.input[0] = vec.output.df
        #        dis.input.array = vec.output.array
        cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
        #        cnt.input[0] = dis.output.dist
        cnt.input[0] = vec.output.result
        global times
        times = 0
        s.start()
        _ = vec.result
        # print(table)

    #        computed = dis.dist()
    #        self.assertEquals(computed.shape[0], len(df))
    #        truth = pairwise_distances(vec.toarray(), metric=dis._metric)
    #        self.assertTrue(np.allclose(truth, computed))

    def test_csv_distances(self) -> None:
        s = self.scheduler()
        vec = CSVLoader(
            get_dataset("smallfile"), index_col=False, header=None, scheduler=s
        )
        #        dis=PairwiseDistances(metric='euclidean',scheduler=s)
        #        dis.input[0] = vec.output.df
        cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
        #        cnt.input[0] = dis.output.dist
        cnt.input[0] = vec.output.result
        global times
        times = 0
        aio.run(s.start(ten_times))
        _ = vec.result
        # print(repr(table))


#        computed = dis.dist()
# self.assertEquals(computed.shape[0], len(df))

#        offset=0
#        size=offset+5000
#        truth = pairwise_distances(df.iloc[offset:size], metric=dis._metric)
#        dist = computed[offset:size,offset:size]
#        self.assertTrue(np.allclose(truth, dist,atol=1e-7)) # reduce tolerance

if __name__ == "__main__":
    ProgressiveTest.main()
