import numpy as np

from ..core.module import ReturnRunStep
from ..core import Sink
from ..stats.kll import KLLSketch
from ..table.module import TableModule
from ..table.table import Table
from ..core.slot import SlotDescriptor
from ..core.decorators import process_slot, run_if_any


class StatsFactory(TableModule):
    """
    Adds statistics on input data
    """

    inputs = [
        SlotDescriptor("table", type=Table, required=True),
    ]

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context is not None
        with self.context as ctx:
            dfslot = ctx.table
            input_df = dfslot.data()
            if not input_df or len(input_df) == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            source_m = dfslot.output_module
            scheduler = self.scheduler()
            with scheduler:
                for col in input_df.columns:
                    if input_df[col].dtype.char not in (
                        np.typecodes["AllInteger"] + np.typecodes["AllFloat"]
                    ):
                        continue
                    print("adding stats on", col)
                    kll = KLLSketch(column=col, scheduler=scheduler)
                    sink = Sink(scheduler=scheduler)
                    kll.input.table = source_m.output.result
                    sink.input.inp = kll.output.result
            return self._return_terminate()
