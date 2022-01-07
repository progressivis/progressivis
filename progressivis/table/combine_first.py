from __future__ import annotations

import logging

import numpy as np

from .nary import NAry, ReturnRunStep
from .table import BaseTable, Table
from .dshape import dshape_union
from ..core.bitmap import bitmap

from typing import List


logger = logging.getLogger(__name__)


def combine_first(table: BaseTable, other: BaseTable, name: str = None) -> Table:
    dshape = dshape_union(table.dshape, other.dshape)
    comb_table = Table(name=name, dshape=dshape)
    if np.all(table.index == other.index):  # the gentle case
        comb_table.resize(len(table.index), index=table.index)
        for cname in table.columns:
            comb_table.loc[:, [cname]] = table.loc[:, [cname]]
            if cname in other.columns:
                nans = bitmap(np.nonzero(np.isnan(table._column(cname).values))[0])
                comb_table.loc[nans, [cname]] = other.loc[nans, [cname]]
        for cname in other.columns:
            if cname in table.columns:
                continue
            comb_table.loc[:, [cname]] = other.loc[:, [cname]]
    else:
        self_set = bitmap(table.index)
        other_set = bitmap(other.index)
        comb_idx = self_set | other_set
        common_set = self_set & other_set
        # common_idx = sorted(common_set)
        self_u_common_idx = self_set | common_set
        other_u_common_idx = other_set | common_set
        other_only_idx = other_set - self_set
        comb_table.resize(len(comb_idx), index=comb_idx)
        for cname in table.columns:
            comb_table.loc[self_u_common_idx, [cname]] = table.loc[
                self_u_common_idx, [cname]
            ]
            if cname in other.columns:
                nans = table.index[np.isnan(table._column(cname).values)]
                snans = nans & common_set
                comb_table.loc[snans, [cname]] = other.loc[snans, [cname]]
            comb_table.loc[other_only_idx, [cname]] = other.loc[other_only_idx, [cname]]
        for cname in other.columns:
            if cname in table.columns:
                continue
            comb_table.loc[other_u_common_idx, [cname]] = other.loc[
                other_u_common_idx, [cname]
            ]
    return comb_table


class CombineFirst(NAry):
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        logger.debug("Entering CombineFirst::run_step")
        frames: List[BaseTable] = []
        for name in self.get_input_slot_multiple():
            slot = self.get_input_slot(name)
            slot.clear_buffers()
            df = slot.data()
            frames.append(df)
        df = frames[0]
        for other in frames[1:]:
            df = combine_first(df, other)
        steps = len(df)
        if self.result is not None:
            self.result = None  # TableModule does not want to reassign result
        self.result = df
        return self._return_run_step(self.state_blocked, steps_run=steps)
