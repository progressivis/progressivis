from . import ProgressiveTest

from progressivis import Every
from progressivis.io import CSVLoader
from progressivis.stats import Histogram2D, Min, Max
from progressivis.vis import Heatmap
from progressivis.datasets import get_dataset


class TestHistogram2D(ProgressiveTest):

    def test_histogram2d(self):
        s = self.scheduler()
        csv = CSVLoader(get_dataset('bigfile'),
                        index_col=False,
                        header=None,
                        scheduler=s)
        min_ = Min(scheduler=s)
        min_.input.table = csv.output.table
        max_ = Max(scheduler=s)
        max_.input.table = csv.output.table
        histogram2d = Histogram2D(1, 2, xbins=100, ybins=100,
                                  scheduler=s)  # columns are called 1..30
        histogram2d.input.table = csv.output.table
        histogram2d.input.min = min_.output.table
        histogram2d.input.max = max_.output.table
        heatmap = Heatmap(filename='histo_%03d.png', scheduler=s)
        heatmap.input.array = histogram2d.output.table
        # pr = Print(scheduler=s)
        pr = Every(proc=self.terse, scheduler=s)
        # pr.input.df = heatmap.output.heatmap
        # pr.input.df = histogram2d.output.df
        pr.input.df = csv.output.table
        csv.scheduler().start()
        s.join()
        s = histogram2d.trace_stats()


if __name__ == '__main__':
    ProgressiveTest.main()
