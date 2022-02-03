import logging
from ..core import Sink, Scheduler
from ..stats.kll import KLLSketch
from ..table.module import TableModule
from ..table.table import Table
from ..core.slot import SlotDescriptor
from ..core.decorators import process_slot, run_if_any

from typing import Callable

logger = logging.getLogger(__name__)


def make_col_stats(imod: TableModule, col: str) -> Callable:
    async def _col_stats(scheduler: Scheduler, run_number: int) -> None:
        with scheduler:
            print("adding stats on", col)
            kll = KLLSketch(column=col, scheduler=scheduler)
            sink = Sink(scheduler=scheduler)
            kll.input.table = imod.output.result
            sink.input.inp = kll.output.result

    return _col_stats


class StatsFactory(TableModule):
    """
    Adds statistics on input data
    """

    inputs = [
        SlotDescriptor("table", type=Table, required=True),
    ]

    def __init__(self, **kwds):
        """ """
        super().__init__(**kwds)
        pass

    def reset(self):
        pass

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            dfslot = ctx.table
            input_df = dfslot.data()
            if not input_df:
                return self._return_run_step(self.state_blocked, steps_run=0)
            source_m = dfslot.output_module
            scheduler = self.scheduler()
            for col in input_df.columns:
                coro = make_col_stats(source_m, col)
                scheduler.on_loop(coro, 1)
            return self._return_run_step(self.state_zombie, steps_run=0)

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step_OK(self, run_number, step_size, howlong):
        with self.context as ctx:
            dfslot = ctx.table
            input_df = dfslot.data()
            if not input_df:
                return self._return_run_step(self.state_blocked, steps_run=0)
            source_m = dfslot.output_module
            scheduler = self.scheduler()
            for i, col in enumerate(input_df.columns):
                coro = make_col_stats(source_m, col)
                scheduler.on_loop(coro, i + 1)
            return self._return_run_step(self.state_zombie, steps_run=0)
