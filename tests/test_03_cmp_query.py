from __future__ import annotations

import pandas as pd

from . import ProgressiveTest
from progressivis.core import aio
from progressivis import Print
from progressivis.table import PTable
from progressivis.table.cmp_query import CmpQueryLast
from progressivis.table.constant import Constant
from progressivis.stats import RandomPTable
from progressivis.table.stirrer import Stirrer
from progressivis.core.pintset import PIntSet

from typing import Any


class TestCmpQuery(ProgressiveTest):
    def test_cmp_query(self) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10000, scheduler=s)
        cmp_ = CmpQueryLast(scheduler=s)
        cst = PTable("cmp_table", data={"_1": [0.5]})
        value = Constant(cst, scheduler=s)
        cmp_.input.cmp = value.output.result
        cmp_.input[0] = random.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = cmp_.output.select
        aio.run(s.start())
        tbl = cmp_.get_input_slot("table").data()
        df = pd.DataFrame(tbl.to_dict(), index=tbl.index.to_array())
        dfe = df.eval("_1<0.5")
        self.assertEqual(cmp_.select, PIntSet(df.index[dfe]))
        # s.join()

    def t_cmp_query_impl(self, **kw: Any) -> None:
        s = self.scheduler()
        random = RandomPTable(10, rows=10000, scheduler=s)
        stirrer = Stirrer(update_column="_1", fixed_step_size=100, scheduler=s, **kw)
        stirrer.input[0] = random.output.result
        cmp_ = CmpQueryLast(scheduler=s)
        cst = PTable("cmp_table", data={"_1": [0.5]})
        value = Constant(cst, scheduler=s)
        cmp_.input.cmp = value.output.result
        cmp_.input[0] = stirrer.output.result
        pr = Print(proc=self.terse, scheduler=s)
        pr.input[0] = cmp_.output.select
        aio.run(s.start())
        tbl = cmp_.get_input_slot("table").data()
        df = pd.DataFrame(tbl.to_dict(), index=tbl.index.to_array())
        dfe = df.eval("_1<0.5")
        self.assertEqual(cmp_.select, PIntSet(df.index[dfe]))

    def test_cmp_query2(self) -> None:
        return self.t_cmp_query_impl(delete_rows=5)

    def test_cmp_query3(self) -> None:
        return self.t_cmp_query_impl(update_rows=5)


if __name__ == "__main__":
    ProgressiveTest.main()
