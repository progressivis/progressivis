from . import ProgressiveTest

import pandas as pd

from progressivis.core import aio
from progressivis import Print
from progressivis.stats.histogram1d_categorical import Histogram1DCategorical
from progressivis.io import SimpleCSVLoader
from progressivis.datasets import get_dataset


class TestHistogram1DCategorical(ProgressiveTest):
    def test_h1d_cat(self) -> None:
        s = self.scheduler()
        random = SimpleCSVLoader(
            get_dataset("bigfile_multiscale"), nrows=10_000, scheduler=s
        )
        h1d_cat = Histogram1DCategorical(column="S", scheduler=s)
        h1d_cat.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = h1d_cat.output.result
        aio.run(s.start())
        assert random.result is not None
        column = random.result["S"]
        valcounts = pd.Series(column.values).value_counts().to_dict()
        print(valcounts)
        self.assertEqual(h1d_cat.result, valcounts)


if __name__ == "__main__":
    ProgressiveTest.main()
