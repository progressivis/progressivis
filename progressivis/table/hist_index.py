"""Histogram Index computes a index for numerical values by
maintaining multiple bitmaps for value ranges, each bitmap corresponds
to a value bin.  The first bin corresponds to half infinite values
lower than the first specifid value, and the last bin corresponds to
half infinite values higher than the last specified value.
"""
from __future__ import absolute_import, division, print_function

import operator
import logging

import numpy as np

from progressivis.core.bitmap import bitmap
from progressivis.core.slot import SlotDescriptor
from progressivis.core.utils import (slice_to_arange, fix_loc, indices_to_slice)
from progressivis.stats import Min, Max
from .module import TableModule
from . import Table
from . import TableSelectedView


logger = logging.getLogger(__name__)


class _HistogramIndexImpl(object):
    def __init__(self, column, table, e_min, e_max, nb_bin):
        #self.table = table
        self.column = table[column] #  column
        self.bins = nb_bin
        self.e_min = e_min
        self.e_max = e_max
        self.bitmaps = None
        self.bins = None
        self._init_histogram(e_min, e_max, nb_bin)

    def _init_histogram(self, e_min, e_max, nb_bin):
        self.bins = np.linspace(e_min, e_max, nb_bin, endpoint=True)
        assert len(self.bins) == nb_bin
        self.bitmaps = [bitmap() for _ in range(nb_bin+1)]

    def reshape(self, min_, max_):
        "Change the bounds of the index if needed"
        pass  # to be defined...then implemented

    def _get_bin(self, val):
        i = np.digitize(val, self.bins)
        return self.bitmaps[int(i)]

    def update_histogram(self, created, updated, deleted):
        "Update the histogram index"
        created = bitmap.asbitmap(created)
        updated = bitmap.asbitmap(updated)
        deleted = bitmap.asbitmap(deleted)
        if deleted or updated:
            to_remove = updated | deleted
            for bm in self.bitmaps:
                bm -= to_remove
        if created or updated:
            to_add = created | updated
            ids = np.array(to_add, np.int64)
            values = self.column.loc[to_add]
            bins = np.digitize(values, self.bins)
            counts = np.bincount(bins)
            for i in np.nonzero(counts)[0]:
                bm = self.bitmaps[i]
                selection = (bins == i)    # boolean mask of values in bin i
                bm.update(ids[selection])  # add them to the bitmap

    def query(self, operator_, limit, approximate=False):  # blocking...
        """
        Return the list of rows matching the query.
        For example, returning all values less than 10 (< 10) would be
        `query(operator.__lt__, 10)`
        """
        pos = np.digitize(limit, self.bins)
        detail = bitmap()
        if not approximate:
            ids = np.array(self.bitmaps[pos], np.int64)
            values = self.column.loc[ids]
            selected = ids[operator_(values, limit)]
            detail.update(selected)

        if operator_ in (operator.lt, operator.le):
            for bm in self.bitmaps[:pos]:
                detail.update(bm)
        else:
            for bm in self.bitmaps[pos + 1:]:
                detail.update(bm)
        return detail

    def range_query(self, lower, upper, approximate=False):
        """
        Return the list of rows with values in range [`lower`, `upper`[
        """
        if lower > upper:
            lower, upper = upper, lower
        pos = np.digitize([lower, upper], self.bins)
        detail = bitmap()
        if not approximate:
            ids = np.array(self.bitmaps[pos[0]], np.int64)
            values = self.column.loc[ids]
            if pos[0] == pos[1]:
                selected = ids[lower <= values < upper]
            else:
                selected = ids[lower <= values]
                detail.update(selected)
                ids = np.array(self.bitmaps[pos[1]], np.int64)
                selected = ids[values < upper]
                detail.update(selected)
        for bm in self.bitmaps[pos[0] + 1:pos[1]]:
            detail.update(bm)
        return detail


class HistogramIndex(TableModule):
    """
    Compute and maintain an histogram index
    """
    parameters = [
        ('bins', np.dtype(int), 126), # actually 128 with "-inf" and "inf"
        ('init_threshold', int, 1000),
    ]

    def __init__(self, column, scheduler=None, **kwds):
        self._add_slots(kwds, 'input_descriptors', [
            SlotDescriptor('table', type=Table, required=True),
            SlotDescriptor('min', type=Table, required=True),
            SlotDescriptor('max', type=Table, required=True)
        ])
        super(HistogramIndex, self).__init__(scheduler=scheduler, **kwds)
        self.column = column
        self._impl = None  # will be created when the init_threshold is reached
        self.selection = bitmap() # will be filled when the table is read
        # so realistic initial values for min and max were available
        self.input_module = None
        self.input_slot = None

    def get_bounds(self, min_slot, max_slot):
        "Return the bounds of the input table according to the min and max modules"
        min_slot.created.next()
        with min_slot.lock:
            min_df = min_slot.data()
            if min_df is None:
                return None
            min_ = min_df.last(self.column)

        max_slot.created.next()
        with max_slot.lock:
            max_df = max_slot.data()
            if max_df is None:
                return None
            max_ = max_df.last(self.column)
        return (min_, max_)

    def run_step(self, run_number, step_size, howlong):
        input_slot = self.get_input_slot('table')
        input_slot.update(run_number)
        steps = 0
        with input_slot.lock:
            input_table = input_slot.data()
        if len(input_table) < self.params.init_threshold:
            # there are not enough rows. it's not worth building an index yet
            return self._return_run_step(self.state_blocked, steps_run=0)
        min_slot = self.get_input_slot('min')
        min_slot.update(run_number)
        max_slot = self.get_input_slot('max')
        max_slot.update(run_number)
        bounds = self.get_bounds(min_slot, max_slot)
        if bounds is None:
            logger.debug('No bounds yet at run %d', run_number)
            return self._return_run_step(self.state_blocked, steps_run=0)
        bound_min, bound_max = bounds
        if self._impl is None:
            self._impl = _HistogramIndexImpl(self.column,
                                             input_table,
                                             bound_min, bound_max,
                                             self.params.bins)
            self._table = TableSelectedView(input_table, self.selection)
        else:
            # Many not always, or should the implementation decide?
            self._impl.reshape(bound_min, bound_max)
        deleted = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(as_slice=False)
            #steps += indices_len(deleted) # deleted are constant time
            steps = 1
            self.selection -= deleted
        created = None
        if input_slot.created.any():
            created = input_slot.created.next(step_size, as_slice=False)
            steps += len(created)
            self.selection |= created
        updated = None
        if input_slot.updated.any():
            updated = input_slot.updated.next(step_size, as_slice=False)
            steps += len(updated)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        with input_slot.lock:
            input_table = input_slot.data()
            #self._table = input_table
        self._impl.update_histogram(created, updated, deleted)
        return self._return_run_step(
            self.next_state(input_slot), steps_run=steps)

    def _eval_to_ids(self, operator_, limit, input_ids=None):
        input_slot = self.get_input_slot('table')
        table_ = input_slot.data()
        if input_ids is None:
            input_ids = table_.index
        else:
            input_ids = fix_loc(input_ids)
        x = table_[self.column].loc[input_ids]
        mask_ = operator_(x, limit)
        arr = slice_to_arange(input_ids)
        return bitmap(arr[np.nonzero(mask_)[0]])

    def query(self, operator_, limit, approximate=False):
        """
        Return the list of rows matching the query.
        For example, returning all values less than 10 (< 10) would be
        `query(operator.__lt__, 10)`
        """
        if self._impl:
            return self._impl.query(operator_, limit, approximate)
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return self._eval_to_ids(operator_, limit)

    def range_query(self, lower, upper, approximate=False):
        """
        Return the list of rows with values in range [`lower`, `upper`[
        """
        if self._impl:
            return self._impl.range_query(lower, upper, approximate)
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return (self._eval_to_ids(operator.__lt__, upper) &  # optimize later
                self._eval_to_ids(operator.__ge__, lower))

    def create_dependent_modules(self, input_module, input_slot, **kwds):
        s = self.scheduler()
        self.input_module = input_module
        self.input_slot = input_slot
        min_ = Min(group=self.id, scheduler=s)
        max_ = Max(group=self.id, scheduler=s)
        min_.input.table = input_module.output[input_slot]
        max_.input.table = input_module.output[input_slot]
        hist_index = self
        hist_index.input.table = input_module.output[input_slot]
        hist_index.input.min = min_.output.table
        hist_index.input.max = max_.output.table
        return hist_index
