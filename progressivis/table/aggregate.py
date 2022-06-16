from __future__ import annotations

# import logging
from ..table.module import TableModule, ReturnRunStep
from ..core.slot import SlotDescriptor, Slot
from . import Table
from ..stats.utils import (
    OnlineFunctor,
    OnlineMean,
    OnlineVariance,
    OnlineStd,
    OnlineSum,
)
from progressivis.core.utils import indices_len
from ..core.decorators import process_slot, run_if_any
from .dshape import dshape_from_dict
from .group_by import GroupBy, DateTime
from typing import cast, List, Union, Any, Dict, Tuple, Type

# See also : https://arrow.apache.org/docs/python/compute.html#py-grouped-aggrs


class Aggregate(TableModule):
    inputs = [SlotDescriptor("table", type=Table, required=True)]
    registry: Dict[str, Type[OnlineFunctor]] = dict(
        mean=OnlineMean, variance=OnlineVariance, stddev=OnlineStd, sum=OnlineSum
    )

    def __init__(
        self, compute: List[Tuple[str, Union[str, Type[OnlineFunctor]]]], **kwds: Any
    ) -> None:
        super().__init__(**kwds)
        self._compute = [
            (n, self.registry[f] if isinstance(f, str) else f) for (n, f) in compute
        ]
        self._aggr_cols = {f"{n}_{f.name}": (n, f) for (n, f) in self._compute}
        self._by_cols: Any = None
        self._local_index: Dict[Any, Any] = {}
        self._table_index: Dict[Any, int] = {}

    def reset(self) -> None:
        if self.result is not None:
            self.table.resize(0)
        self._local_index = {}
        self._table_index = {}

    def update_row(self, grp, grp_ids, input_df):
        if grp not in self._local_index:
            row_dict = {
                nm: construct[1]() for (nm, construct) in self._aggr_cols.items()
            }
            self._local_index[grp] = row_dict
        else:
            row_dict = self._local_index[grp]
        for nm, computer in row_dict.items():
            col = self._aggr_cols[nm][0]
            computer.add(input_df[col].loc[grp_ids])
        by = self._by_cols
        by_stuff = (
            [(by[0], grp)] if len(by) == 1 else [(b, g) for (b, g) in zip(by, grp)]
        )
        row = dict(
            by_stuff
            + [(col, computer.get_value()) for (col, computer) in row_dict.items()]
        )
        if self.result is None:
            self.result = Table(
                name=self.generate_table_name("aggr"),
                dshape=dshape_from_dict({k: [v] for (k, v) in row.items()}),
                create=True,
            )
        if grp not in self._table_index:
            id = self.table.add(row)
            self._table_index[grp] = id
        else:
            id = self._table_index[grp]
            self.table.loc[id, row_dict.keys()] = row

    @process_slot("table", reset_cb="reset")
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
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            input_df = dfslot.data()
            if input_df is None:
                return self._return_run_step(self.state_blocked, steps_run=0)
            groupby_mod = cast(GroupBy, dfslot.output_module)
            if self._by_cols is None:
                by = groupby_mod.by
                if isinstance(by, str):
                    self._by_cols = [by]
                elif isinstance(by, list):
                    self._by_cols = by
                elif isinstance(by, DateTime):
                    self._by_cols = [f"{by.column}_{by.mask}"]
                elif callable(by):
                    self._by_cols = ["by_col"]
                else:
                    raise ValueError(f"Not implemented {by} type")
            for grp, ids in groupby_mod.items():
                grp_ids = ids & indices
                self.update_row(grp, grp_ids, input_df)
        return self._return_run_step(self.state_ready, steps_run=steps)
