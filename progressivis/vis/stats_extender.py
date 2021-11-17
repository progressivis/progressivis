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

class StatsExtender(TableModule):
    """
    Adds statistics on input data
    """
    parameters = []

    inputs = [SlotDescriptor('table', type=Table, required=True),
              SlotDescriptor('min', type=PsDict, required=False),
              SlotDescriptor('max', type=PsDict, required=False),
              SlotDescriptor('var', type=PsDict, required=False),
              SlotDescriptor('distinct', type=PsDict, required=False),
              SlotDescriptor('corr', type=PsDict, required=False),
    ]
    outputs = [
        SlotDescriptor('dshape', type=PsDict, required=False)
    ]


    def __init__(self, usecols=None, **kwds):
        """
        usecols: these columns are transmitted to stats modules by default
        """
        super().__init__(**kwds)
        self._usecols = usecols
        self.visible_cols = []
        self.decorations = []
        self._raw_dshape = None
        self._dshape = None
        self._dshape_flag = False

    def reset(self):
        if self.result is not None:
            self.result.selection = bitmap()

    def starting(self):
        super().starting()
        ds_slot = self.get_output_slot('dshape')
        if ds_slot:
            logger.debug('Maintaining dshape')
            self._dshape_flag = True
        else:
            self._dshape_flag = False

    def maintain_dshape(self, data):
        if not self._dshape_flag or self._raw_dshape == data.dshape:
            return
        self._raw_dshape = data.dshape
        if self._dshape is None:
            self._dshape = PsDict()
        self._dshape.update(dshape_fields(data.dshape))

    def get_data(self, name):
        if name == 'dshape':
            return self._dshape
        return super().get_data(name)

    @property
    def dshape(self):
        return self._dshape

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            dfslot = ctx.table
            input_df = dfslot.data()
            if not input_df:
                return self._return_run_step(self.state_blocked, steps_run=0)
            cols = self._columns or input_df.columns
            usecols = self._usecols or cols
            self.visible_cols = usecols
            for name_ in self.input_slot_names():
                if name_ == "table":
                    continue
                dec_slot = self.get_input_slot(name_)
                if dec_slot and dec_slot.has_buffered():
                    dec_slot.clear_buffers()
            indices = dfslot.created.next(step_size, as_slice=False)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            self.maintain_dshape(input_df)
            if self.result is None:
                self.result = TableSelectedView(input_df, bitmap([]))
            self.result.selection |= indices
            return self._return_run_step(self.next_state(dfslot), steps)

    def _get_usecols(self, x):
        return x if isinstance(x, Iterable) else self._usecols

    def _get_usecols_hist(self, x, hist):
        if not hist:
            return self._get_usecols(x)
        if not x:
            return self._get_usecols(hist)
        if isinstance(x, Iterable) and isinstance(hist, Iterable):
            return list(set(list(x)+list(hist)))
        return self._usecols

    def create_dependent_modules(self, input_module, input_slot='result',
                                 min_=False,
                                 max_=False,
                                 hist=False,
                                 var=False,
                                 distinct=False,
                                 corr=False,
                                 dshape=True
    ):
        s = self.scheduler()
        self.input.table = input_module.output[input_slot]
        usecols = self._usecols
        if min_ or hist:
            self.min = Min(scheduler=s,
                           columns=self._get_usecols_hist(min_, hist))
            self.min.input.table = input_module.output[input_slot]
            self.input.min = self.min.output.result
            self.decorations.append('min')
        if max_ or hist:
            self.max = Max(scheduler=s, columns=self._get_usecols_hist(max_, hist))
            self.max.input.table = input_module.output[input_slot]
            self.input.max = self.max.output.result
            self.decorations.append('max')
        self.hist = {}
        if hist:
            for col in self._get_usecols(hist):
                hist1d = Histogram1D(scheduler=s, column=col)
                hist1d.input.table = input_module.output[input_slot]
                hist1d.input.min = self.min.output.result
                hist1d.input.max = self.max.output.result
                pr = Print(proc=lambda x: None, scheduler=s)
                pr.input[0] = hist1d.output.categorical
                self.hist[col] = hist1d
        if var:
            self.var = Var(scheduler=s,
                           columns=self._get_usecols(var),
                           ignore_string_cols=True)
            self.var.input.table = input_module.output[input_slot]
            self.input.var = self.var.output.result
            self.decorations.append('var')
        if distinct:
            self.distinct = Distinct(scheduler=s, columns=self._get_usecols(distinct))
            self.distinct.input.table = input_module.output[input_slot]
            self.input.distinct = self.distinct.output.result
            self.decorations.append('distinct')
        if corr:
            self.corr = Corr(scheduler=s,
                             columns=self._get_usecols(corr),
                             ignore_string_cols=True)
            self.corr.input.table = input_module.output[input_slot]
            self.input.corr = self.corr.output.result

        if dshape:
            pr = Print(proc=lambda x: None, scheduler=s)
            pr.input[0] = self.output.dshape
