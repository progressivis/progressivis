import numpy as np
import pandas as pd
import logging
from collections import Iterable

from ..core import Print
from ..core.bitmap import bitmap
from ..core.utils import indices_len, fix_loc
from ..core.slot import SlotDescriptor
from ..table.table import Table
from ..table.module import TableModule
from ..table.nary import NAry
from ..utils.psdict import PsDict
#from .var import OnlineVariance
from ..stats import Min, Max, Var, Distinct, Histogram1D, Corr
from ..core.decorators import process_slot, run_if_any
from ..table import TableSelectedView
from ..table.dshape import dshape_fields

logger = logging.getLogger(__name__)

class Switch(TableModule):
    """
    Select the output (result or result_else) ar runtime
    """
    parameters = []

    inputs = [
        SlotDescriptor('table', type=Table, required=True)
    ]
    outputs = [
        SlotDescriptor('result_else', type=Table, required=False)
    ]


    def __init__(self, condition, **kwds):
        """
        condition: callable which should return
        * None => undecidable (yet), run_step must return blocked_state
        * True => run_step output is 'result'
        * False => run_step output is 'result_else'
        """
        assert callable(condition)
        super().__init__(**kwds)
        self._condition = condition
        self.result_else = None
        self._output = None

    def reset(self):
        if self.result is not None:
            self.result.selection = bitmap()
        if self.result_else is not None:            
            self.result_else.selection = bitmap()            

    def get_data(self, name):
        if name == 'result_else':
            return self.result_else
        return super().get_data(name)

    def run_step(self, run_number, step_size, howlong):
        slot = self.get_input_slot('table')
        input_df = slot.data()
        if input_df is None or not slot.has_buffered():
            return self._return_run_step(self.state_blocked, steps_run=0)
        cond = self._condition(self)
        if cond is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self._output is None:
            if cond:
                self._output = \
                self.result = \
                TableSelectedView(input_df, bitmap([]))
            else:
                self._output = \
                self.result_else = \
                TableSelectedView(input_df, bitmap([]))
        steps = 0
        created_ids = bitmap()
        if slot.created.any():
            created_ids = slot.created.next(as_slice=False)
            steps += indices_len(created_ids)
            self._output.selection |= created_ids
        updated_ids = bitmap()
        if slot.updated.any():
            updated_ids = slot.updated.next(as_slice=False)
            steps += indices_len(updated_ids)
            print("Updates are ignored in switch")
            #self._output._base.add_updated(updated_ids)
        deleted_ids = bitmap()
        if slot.deleted.any():
            deleted_ids = slot.deleted.next(as_slice=False)
            steps += indices_len(deleted_ids)
            self._output.selection -= deleted_ids
        return self._return_run_step(self.next_state(slot),
                                     steps_run=steps)
