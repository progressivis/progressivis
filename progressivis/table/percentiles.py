import numpy as np

from . import Table
from . import TableSelectedView
from ..core.slot import SlotDescriptor
from .module import TableModule
from collections import OrderedDict
from ..utils.psdict import PsDict


class Percentiles(TableModule):
    parameters = [('accuracy', np.dtype(float), 0.5)]
    inputs = [SlotDescriptor('table', type=Table, required=True),
              SlotDescriptor('percentiles', type=PsDict, required=True)]

    def __init__(self, hist_index, **kwds):
        super(Percentiles, self).__init__(**kwds)
        self._accuracy = self.params.accuracy
        self._hist_index = hist_index
        self.default_step_size = 1000

    def compute_percentiles(self, points, input_table):
        column = input_table[self._hist_index.column]
        hii = self._hist_index._impl

        def _filter_tsv(bm):
            return bm & input_table.selection

        def _no_filtering(bm):
            return bm
        _filter = _filter_tsv if isinstance(input_table, TableSelectedView) else _no_filtering
        len_ = len(input_table)
        k_points = [p*(len_+1)*0.01 for p in points.values()]
        max_k = max(k_points)
        ret_values = []
        k_accuracy = self._accuracy * len_ * 0.01
        acc = 0
        lbm = len(hii.bitmaps)
        acc_list = np.empty(lbm, dtype=np.int64)
        sz_list = np.empty(lbm, dtype=np.int64)
        bm_list = []
        for i, bm in enumerate(hii.bitmaps):
            fbm = _filter(bm)
            sz = len(fbm)
            acc += sz
            sz_list[i] = sz
            acc_list[i] = acc
            bm_list.append(fbm)
            if acc > max_k:
                break  # just avoids unnecessary computes
        acc_list = acc_list[:i+1]
        for k in k_points:
            i = (acc_list >= k).nonzero()[0][0]
            reminder = int(acc_list[i] - k)
            assert sz_list[i] > reminder >= 0
            if sz_list[i] < k_accuracy:
                ret_values.append(column[bm_list[i][0]])
            else:
                values = column.loc[bm_list[i]]
                part = np.partition(values, reminder)
                ret_values.append(values[reminder])
        return OrderedDict(zip(points.keys(), ret_values))

    def run_step(self, run_number, step_size, howlong):
        input_slot = self.get_input_slot('table')
        if input_slot.data() is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        input_slot.update(run_number)
        steps = 0
        if input_slot.deleted.any():
            input_slot.deleted.next(step_size)
            steps = 1
        if input_slot.created.any():
            input_slot.created.next(step_size)
            steps = 1
        if input_slot.updated.any():
            input_slot.updated.next(step_size)
            steps = 1
        # with input_slot.lock:
        #     input_table = input_slot.data()
        # param = self.params
        percentiles_slot = self.get_input_slot('percentiles')
        if percentiles_slot.data() is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        percentiles_slot.update(run_number)
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
        if not self._hist_index._impl:
            return self._return_run_step(self.state_blocked, steps_run=0)
        computed = self.compute_percentiles(
            percentiles_slot.data(),
            input_slot.data())
        if not self._table:
            self._table = Table(name=None,
                                dshape=percentiles_slot.data().dshape)
            self._table.add(computed)
        else:
            self._table.loc[0, :] = list(computed.values())
        return self._return_run_step(self.next_state(input_slot), steps)
