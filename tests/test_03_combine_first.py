from . import ProgressiveTest
from progressivis import Print
from progressivis.table.constant import Constant
from progressivis.table.combine_first import CombineFirst
from progressivis.table.table import Table
from progressivis.core import aio


import pandas as pd
import numpy as np


class TestCombineFirst(ProgressiveTest):
    def test_combine_first_dup(self) -> None:
        s = self.scheduler(True)
        cst1 = Constant(
            Table(
                name="tcf_xmin_xmax",
                data=pd.DataFrame({"xmin": [1], "xmax": [2]}),
                create=True,
            ),
            scheduler=s,
        )
        cst2 = Constant(
            Table(
                name="tcf_ymin_ymax",
                data=pd.DataFrame({"ymin": [5], "ymax": [6]}),
                create=True,
            ),
            scheduler=s,
        )
        cst3 = Constant(
            Table(
                name="tcf_ymin_ymax2",
                data=pd.DataFrame({"ymin": [3], "ymax": [4]}),
                create=True,
            ),
            scheduler=s,
        )
        cf = CombineFirst(scheduler=s)
        cf.input[0] = cst1.output.result
        cf.input[0] = cst2.output.result
        cf.input[0] = cst3.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = cf.output.result
        aio.run(s.start())
        # res = cf.trace_stats(max_runs=1)
        df = cf.table
        row = df.last()
        self.assertFalse(row is None)
        assert row is not None
        last = row.to_dict()
        self.assertEquals(last["xmin"], 1)
        self.assertEquals(last["xmax"], 2)
        self.assertEquals(last["ymin"], 5)
        self.assertEquals(last["ymax"], 6)

    def test_combine_first_nan(self) -> None:
        s = self.scheduler(True)
        cst1 = Constant(
            Table(
                name="tcf_xmin_xmax_nan",
                data=pd.DataFrame({"xmin": [1], "xmax": [2]}),
                create=True,
            ),
            scheduler=s,
        )
        cst2 = Constant(
            Table(
                name="tcf_ymin_ymax_nan",
                data=pd.DataFrame({"ymin": [np.nan], "ymax": [np.nan]}),
                create=True,
            ),
            scheduler=s,
        )
        cst3 = Constant(
            Table(
                name="tcf_ymin_ymax2_nan",
                data=pd.DataFrame({"ymin": [3], "ymax": [4]}),
                create=True,
            ),
            scheduler=s,
        )
        cf = CombineFirst(scheduler=s)
        cf.input[0] = cst1.output.result
        cf.input[0] = cst2.output.result
        cf.input[0] = cst3.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = cf.output.result
        aio.run(s.start())
        df = cf.table
        row = df.last()
        assert row is not None
        last = row.to_dict()
        self.assertTrue(
            last["xmin"] == 1
            and last["xmax"] == 2
            and last["ymin"] == 3
            and last["ymax"] == 4
        )
