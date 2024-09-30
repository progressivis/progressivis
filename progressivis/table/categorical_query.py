from __future__ import annotations

# import logging
from ..core.module import Module, ReturnRunStep, def_input, def_output, document
from ..core.slot import Slot
from .api import PTable, PTableSelectedView
from ..core.pintset import PIntSet
from ..core.decorators import process_slot, run_if_any
from .group_by import GroupBy
from ..utils.psdict import PDict
from typing import Any, List


@document
@def_input("table", PTable, doc="input data")
@def_input("choice", PDict, doc="provide the subset of categorical values to be queried")
@def_output("result", PTableSelectedView)
class CategoricalQuery(Module):
    """
    Selects rows that contain values in a given column which are part of a provided subset
    It is convenient for categorical data.

    """
    def __init__(self, column: str, choice_key: str = "only", **kwds: Any) -> None:
        """
        Parameters
        ----------
        column:
            filtering column
        choice_key:
            the key in the **choice** input giving the subset to query
        """
        super().__init__(**kwds)
        self._column = column
        self._choice_key = choice_key
        self._only: List[str] = []

    def reset(self) -> None:
        if self.result is not None:
            self.result.selection = PIntSet()

    def create_dependent_modules(
        self,
        input_module: Module,
        input_slot: str = "result",
    ) -> None:
        s = self.scheduler()
        grby = GroupBy(by=self._column, scheduler=s)
        grby.input.table = input_module.output[input_slot]
        self.input.table = grby.output.result
        self.dep.grby = grby

    @process_slot("table", "choice", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            dfslot: Slot = ctx.table
            indices = dfslot.created.next(
                length=step_size, as_slice=False
            )  # returns a slice
            steps = len(indices)
            cat_slot = ctx.choice
            if not cat_slot:
                return self._return_run_step(self.state_blocked, steps_run=0)
            if cat_slot.has_buffered():
                cat_slot.clear_buffers()
                steps += 1
                self._only = cat_slot.data().get(self._choice_key, [])
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_table = dfslot.data()
            if input_table is None:
                return self._return_run_step(self.state_blocked, steps_run=0)
            if self.result is None:
                self.result = PTableSelectedView(input_table, PIntSet([]))
            assert isinstance(dfslot.output_module, GroupBy)
            groupby_mod = dfslot.output_module
            assert self._column == groupby_mod.by
            for grp, ids in groupby_mod.items():
                if grp not in self._only:
                    continue
                grp_ids = ids & indices if indices else ids
                self.result.selection |= grp_ids
                # self.update_row(grp, grp_ids, input_table)
        return self._return_run_step(self.state_ready, steps_run=steps)
