from . import ProgressiveTest, skip

from progressivis import Every, StorageManager, CSVLoader, Histogram1D, Min, Max, get_dataset
from progressivis.core import aio


@skip("Not ready")  # essay
class TestHistogram1D(ProgressiveTest):
    def tearDown(self) -> None:
        StorageManager.default.end()

    def test_histogram1d(self) -> None:
        s = self.scheduler()
        csv = CSVLoader(
            get_dataset("bigfile"),
            force_valid_ids=True,
            header=None,
            scheduler=s,
        )
        min_ = Min(scheduler=s)
        min_.input[0] = csv.output.result
        max_ = Max(scheduler=s)
        max_.input[0] = csv.output.result
        histogram1d = Histogram1D("_2", scheduler=s)  # columns are called 1..30
        histogram1d.input[0] = csv.output.result
        histogram1d.input.min = min_.output.result
        histogram1d.input.max = max_.output.result

        # pr = Print(scheduler=s)
        pr = Every(proc=self.terse, scheduler=s)
        pr.input[0] = csv.output.result
        aio.run(s.start())
        # s = histogram1d.trace_stats()
        # print "Done. Run time: %gs, loaded %d rows" % (s['duration'].irow(-1), len(module.df()))
        # pd.set_option('display.expand_frame_repr', False)
        # print(s)


if __name__ == "__main__":
    ProgressiveTest.main()
