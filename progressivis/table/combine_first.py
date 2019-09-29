from __future__ import absolute_import, division, print_function

import numpy as np

from .nary import NAry
from .table import Table
from .dshape import dshape_union


def combine_first(table, other, name=None):
    dshape = dshape_union(table.dshape, other.dshape)
    comb_table = Table(name=name, dshape=dshape)
    if np.all(table.index.values == other.index.values):  # the gentle case
        comb_table.resize(len(table.index), index=table.index)
        for cname in table.columns:
            comb_table.loc[:, [cname]] = table.loc[:, [cname]]
            if cname in other.columns:
                nans = table.index.values[np.isnan(table._column(cname).values)]
                comb_table.loc[nans, [cname]] = other.loc[nans, [cname]]
        for cname in other.columns:
            if cname in table.columns:
                continue
            comb_table.loc[:, [cname]] = other.loc[:, [cname]]
    else:
        self_set = set(table.index.values)
        other_set = set(other.index.values)
        comb_idx = sorted(self_set | other_set)
        common_set = self_set & other_set
        # common_idx = sorted(common_set)
        self_u_common_idx = sorted(self_set | common_set)
        other_u_common_idx = sorted(other_set | common_set)
        other_only_idx = sorted(other_set - self_set)
        comb_table.resize(len(comb_idx), index=comb_idx)
        for cname in table.columns:
            comb_table.loc[self_u_common_idx, [cname]] = table.loc[self_u_common_idx, [cname]]
            if cname in other.columns:
                nans = table.index.values[np.isnan(table._column(cname).values)]
                nans = sorted(set(nans) & common_set)
                comb_table.loc[nans, [cname]] = other.loc[nans, [cname]]
            comb_table.loc[other_only_idx, [cname]] = other.loc[other_only_idx, [cname]]
        for cname in other.columns:
            if cname in table.columns:
                continue
            comb_table.loc[other_u_common_idx, [cname]] = other.loc[other_u_common_idx, [cname]]
    return comb_table


class CombineFirst(NAry):
    def run_step(self, run_number, step_size, howlong):
        frames = []
        for name in self.get_input_slot_multiple():
            slot = self.get_input_slot(name)
            with slot.lock:
                df = slot.data()
            frames.append(df)
        df = frames[0]
        for other in frames[1:]:
            df = combine_first(df, other)
        steps = len(df)
        with self.lock:
            self._table = df
        return self._return_run_step(self.state_blocked, steps_run=steps)
