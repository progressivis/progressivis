from . import ProgressiveTest, skip, main
from progressivis.core import aio
from progressivis import Print, Scheduler
from progressivis.stats import RandomTable
from progressivis.table.filtermod import FilterMod
from progressivis.core.bitmap import bitmap
from progressivis.table.stirrer import Stirrer
import pandas as pd
import numpy as np

class TestFilter(ProgressiveTest):
    def test_filter(self):
        s = Scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        filter_ = FilterMod(expr='_1 > 0.5', scheduler=s)
        filter_.input.table = random.output.table
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = filter_.output.table
        aio.run(s.start())
        idx = filter_.get_input_slot('table')\
                     .data()\
                     .eval('_1>0.5', result_object='index')
        self.assertEqual(filter_._table.index, bitmap(idx))

    def test_filter2(self):
        s = Scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(update_column='_1', delete_rows=5,
                          #update_rows=5,
                          fixed_step_size=100, scheduler=s)
        stirrer.input.table = random.output.table
        filter_ = FilterMod(expr='_1 > 0.5', scheduler=s)
        filter_.input.table = stirrer.output.table
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = filter_.output.table
        aio.run(s.start())
        tbl = filter_.get_input_slot('table').data()
        idx = tbl.eval('_1>0.5', result_object='index')
        self.assertEqual(filter_._table.index, bitmap(idx))
        df = pd.DataFrame(tbl.to_dict(),index=tbl.index.to_array())
        dfe = df.eval('_1>0.5')
        self.assertEqual(filter_._table.index, bitmap(df.index[dfe]))

    def test_filter3(self):
        s = Scheduler()
        random = RandomTable(2, rows=100000, scheduler=s)
        stirrer = Stirrer(update_column='_1',
                          update_rows=5,
                          fixed_step_size=100, scheduler=s)
        stirrer.input.table = random.output.table
        filter_ = FilterMod(expr='_1 > 0.5', scheduler=s)
        filter_.input.table = stirrer.output.table
        pr = Print(proc=self.terse, scheduler=s)
        pr.input.df = filter_.output.table
        aio.run(s.start())
        tbl = filter_.get_input_slot('table').data()
        idx = tbl.eval('_1>0.5', result_object='index')
        self.assertEqual(filter_._table.index, bitmap(idx))
        df = pd.DataFrame(tbl.to_dict(),index=tbl.index.to_array())
        dfe = df.eval('_1>0.5')
        self.assertEqual(filter_._table.index, bitmap(df.index[dfe]))

if __name__ == '__main__':
    main()
