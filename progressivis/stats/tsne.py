from __future__ import annotations

import pandas as pd
from ..core.api import Slot
from ..core.module import (Module,
                           ReturnRunStep,
                           def_input,
                           def_output,
                           document)
from ..table.api import PTable
from ..core.pintset import PIntSet
from ..table.dshape import dshape_from_dataframe
from ..core.docstrings import INPUT_SEL
from ..core.decorators import process_slot, run_always
try:
    from pynene import ProgressiVisTSNE  # type: ignore
except Exception:
    ProgressiVisTSNE = None

    from typing import Any


@document
@def_input("table", PTable, hint_type=str, doc=INPUT_SEL)
@def_output(
    "result",
    PTable,
    doc=(""),
)
class TSNE(Module):
    """
    Computes the maximum of the values for every column of an input table.
    """

    def __init__(
            self,
            array_col: str,
            output_cols: list[str],
            max_iter: int = 1000,
            qual_lim: float = .0,
            is_greedy: bool = True,
        **kwds: Any,
    ) -> None:
        """
        Args:
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)
        self._array_col = array_col
        self._output_dims = len(output_cols)
        self._output_cols = output_cols
        self._is_greedy: bool = is_greedy
        #self.default_step_size = 10_000
        self._max_iter = max_iter
        self._qual_lim = qual_lim
        self.tsne: ProgressiVisTSNE | None = None

    def reset(self) -> None:
        self.tsne = None
        if self.result is not None:
            self.result.resize(0)

    def is_greedy(self) -> bool:
        return self._is_greedy

    # def todo_to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
    #     json = super().to_json(short, with_speed)
    #     if short:
    #         return json
    #     return self._tsne_to_json(json)

    # def _tsne_to_json(self, json: JSon) -> None:
    #     data = [] if self.tsne is None else self.tsne.get_y().tolist()
    #     json["data"] = data

    @process_slot("table", reset_cb="reset")
    @run_always
    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            dfslot: Slot = ctx.table
            indices = dfslot.created.next(
                as_slice=False  # length=step_size,
            )  # returns a slice
            input_df = dfslot.data()
            if input_df is None:
                return self._return_run_step(self.state_blocked, steps_run=0)
            steps = len(indices)
            #import pdb;pdb.set_trace()
            #if steps == 0:
            #    #indices = input_df.index
            #    steps = len(indices)
            #else:
            #if len(input_df) < 10_000:
            #    #print("PROBLEM", len(input_df))
            #    return self._return_run_step(self.state_blocked, steps_run=steps)
            if self.tsne is None:
                self.tsne = ProgressiVisTSNE(input_df,
                                             self._array_col,
                                             output_dims=self._output_dims)
                indices = PIntSet()
            else:
                if self._max_iter <= 0:
                    self._is_greedy = False
            N = 1 if indices else 5
            self.tsne.run_ids(indices, N)
            self._max_iter -= N
            if self.result is None:
                df = pd.DataFrame(self.tsne.get_y(), columns=self._output_cols)
                self.result = PTable(
                    name=self.generate_table_name("tsne"),
                    dshape=dshape_from_dataframe(df),
                    data=df,
                    create=True
                    )
            else:
                tsne_y = self.tsne.get_y()
                if len(self.result) < len(tsne_y):
                    #import pdb;pdb.set_trace()
                    self.result.resize(len(tsne_y))
                self.result.loc[:, self._output_cols] = tsne_y
            #if self.is_greedy():
            #    return self._return_run_step(self.state_ready, 0)
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)
