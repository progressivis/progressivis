from . import ProgressiveTest, skip

from progressivis import Every
from progressivis.io import CSVLoader
from progressivis.metrics import PairwiseDistances
from progressivis.datasets import get_dataset
from progressivis.core import aio


def print_len(x):
    if x is not None:
        print(x.shape)


times = 0


def ten_times(scheduler, run_number):
    global times
    times += 1
    if times > 10:
        scheduler.stop()


class TestMDS(ProgressiveTest):
    # def test_MDS_vec(self):
    #     vec=VECLoader(get_dataset('warlogs'))
    #     dis=PairwiseDistances(metric='cosine')
    #     dis.input[0] = vec.output.df
    #     dis.input.array = vec.output.array
    #     cnt = Every(proc=print_len,constant_time=True)
    #     cnt.input[0] = dis.output.df
    #     vec.start()

    @skip("Need to implement MDS on tables")
    def test_MDS_csv(self):
        s = self.scheduler()
        vec = CSVLoader(
            get_dataset("smallfile"), index_col=False, header=None, scheduler=s
        )
        dis = PairwiseDistances(metric="euclidean", scheduler=s)
        dis.input[0] = vec.output.df
        cnt = Every(proc=self.terse, constant_time=True, scheduler=s)
        cnt.input[0] = dis.output.dist
        global times
        times = 0
        aio.run(s.start(ten_times))


if __name__ == "__main__":
    ProgressiveTest.main()
