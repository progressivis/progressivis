import logging
from ..core import Sink
from ..stats.kll import KLLSketch
from ..table.module import TableModule
from ..table.table import Table
from ..core.slot import SlotDescriptor
from ..core.decorators import process_slot, run_if_any

logger = logging.getLogger(__name__)


class StatsFactory(TableModule):
    """
    Adds statistics on input data
    """

    inputs = [
        SlotDescriptor("table", type=Table, required=True),
    ]

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            dfslot = ctx.table
            input_df = dfslot.data()
            if not input_df or len(input_df) == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            source_m = dfslot.output_module
            scheduler = self.scheduler()
            with scheduler:
                for col in input_df.columns:
                    # TODO test if col is numerical?
                    print("adding stats on", col)
                    kll = KLLSketch(column=col, scheduler=scheduler)
                    sink = Sink(scheduler=scheduler)
                    kll.input.table = source_m.output.result
                    sink.input.inp = kll.output.result
            return self._return_run_step(self.state_zombie, steps_run=0)
