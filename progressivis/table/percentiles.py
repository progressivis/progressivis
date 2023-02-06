from __future__ import annotations

import numpy as np

from . import PTable, BasePTable
from ..core.pintset import PIntSet
from ..core.module import Module, ReturnRunStep, def_input, def_output, def_parameter
from collections import OrderedDict
from ..utils.psdict import PDict
from .hist_index import HistogramIndex

from typing import Any, Dict, List, cast


@def_parameter("accuracy", np.dtype(float), 0.5)
@def_input("table", PTable)
@def_input("percentiles", PDict)
@def_input("hist", PTable)
@def_output("result", PTable)
class Percentiles(Module):
    """ """

    def __init__(self, **kwds: Any) -> None:
        super(Percentiles, self).__init__(**kwds)
        self._accuracy = self.params.accuracy
        self.default_step_size = 1000

    def compute_percentiles(
        self,
        points: Dict[str, float],
        input_table: BasePTable,
        hist_index: HistogramIndex,
    ) -> Dict[str, float]:
        column = input_table[hist_index.column]
        hii = hist_index._impl
        assert hii is not None

        def _filter_tsv(bm: PIntSet) -> PIntSet:
            return bm & input_table.index

        def _no_filtering(bm: PIntSet) -> PIntSet:
            return bm

        _filter = _filter_tsv if isinstance(input_table, BasePTable) else _no_filtering
        len_ = len(input_table)
        k_points = [p * (len_ + 1) * 0.01 for p in points.values()]
        max_k = max(k_points)
        ret_values: List[float] = []
        k_accuracy = self._accuracy * len_ * 0.01
        acc = 0
        lbm = len(hii.pintsets)
        acc_list = np.empty(lbm, dtype=np.int64)
        sz_list = np.empty(lbm, dtype=np.int64)
        bm_list: List[PIntSet] = []
        for i, bm in enumerate(hii.pintsets):
            fbm = _filter(bm)
            sz = len(fbm)
            acc += sz
            sz_list[i] = sz
            acc_list[i] = acc
            bm_list.append(fbm)
            if acc > max_k:
                break  # just avoids unnecessary computes
        acc_list = acc_list[: i + 1]
        for k in k_points:
            i = (acc_list >= k).nonzero()[0][0]  # type: ignore
            reminder = int(acc_list[i] - k)
            assert sz_list[i] > reminder >= 0
            if sz_list[i] < k_accuracy:
                ret_values.append(column[bm_list[i][0]])
            else:
                values = column.loc[bm_list[i]]
                _ = np.partition(values, reminder)
                ret_values.append(values[reminder])
        return OrderedDict(zip(points.keys(), ret_values))

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        if input_slot.data() is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        # input_slot.update(run_number)
        steps = 0
        if input_slot.deleted.any():
            input_slot.deleted.next(length=step_size)
            steps = 1
        if input_slot.created.any():
            input_slot.created.next(length=step_size)
            steps = 1
        if input_slot.updated.any():
            input_slot.updated.next(length=step_size)
            steps = 1
        # with input_slot.lock:
        #     input_table = input_slot.data()
        # param = self.params
        percentiles_slot = self.get_input_slot("percentiles")
        if percentiles_slot.data() is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        # percentiles_slot.update(run_number)
        percentiles_changed = False
        if percentiles_slot.deleted.any():
            percentiles_slot.deleted.next()
        if percentiles_slot.updated.any():
            percentiles_slot.updated.next()
            percentiles_changed = True
        if percentiles_slot.created.any():
            percentiles_slot.created.next()
            percentiles_changed = True
        if len(percentiles_slot.data()) == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if steps == 0 and not percentiles_changed:
            return self._return_run_step(self.state_blocked, steps_run=0)
        hist_slot = self.get_input_slot("hist")
        hist_slot.deleted.next()
        hist_slot.updated.next()
        hist_slot.created.next()
        hist_index: HistogramIndex = cast(HistogramIndex, hist_slot.output_module)
        if not hist_index._impl:
            return self._return_run_step(self.state_blocked, steps_run=0)
        computed = self.compute_percentiles(
            percentiles_slot.data(), input_slot.data(), hist_index
        )
        table: PTable
        if not self.result:
            table = PTable(name=None, dshape=percentiles_slot.data().dshape)
            table.add(computed)
            self.result = table
        else:
            self.result.loc[0, :] = list(computed.values())
        return self._return_run_step(self.next_state(input_slot), steps)
