from . import ProgressiveTest, main
import asyncio as aio
from progressivis import Print, Scheduler
from progressivis.stats import RandomTable
from progressivis.table.filtermod import FilterMod
from progressivis.core.bitmap import bitmap


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
        self.assertEqual(filter_._table.selection, bitmap(idx))


if __name__ == '__main__':
    main()
