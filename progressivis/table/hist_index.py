"""Histogram Index computes a index for numerical values by
maintaining multiple bitmaps for value ranges, each bitmap corresponds
to a value bin.  The first bin corresponds to half infinite values
lower than the first specifid value, and the last bin corresponds to
half infinite values higher than the last specified value.
"""
from __future__ import absolute_import, division, print_function

from .nary import NAry
from . import Table
from . import TableSelectedView
from ..core.slot import SlotDescriptor
from .module import TableModule

import numpy as np
from bisect import bisect_left, bisect_right

from progressivis.core.utils import (slice_to_arange, slice_to_bitmap,
                                     indices_len, fix_loc, indices_to_slice)
from progressivis.core.bitmap import bitmap
from progressivis.stats import Min, Max

import operator
import functools
from bisect import bisect_right
import logging

import numpy as np

from progressivis.core.utils import (indices_len, fix_loc, slice_to_arange)
from progressivis.core.bitmap import bitmap
from progressivis.stats import Min, Max
from . import Table
from ..core.slot import SlotDescriptor
from .module import TableModule

logger = logging.getLogger(__name__)


class _Bucket(object):
    def __init__(self, left_b, right_b, bm=None):
        self.left_b = left_b
        self.right_b = right_b
        if bm is None:
            self._bitmap = bitmap([])
        else:
            self._bitmap = bm

    @property
    def values(self):
        "Return the bitmap associated with this bucket"
        return self._bitmap

    def __str__(self):
        return "[{}, {})".format(self.left_b, self.right_b)

    def __repr__(self):
        return self.__str__()

    def __lt__(self, x):
        ret = self.right_b < x
        return ret

    def __le__(self, x):
        return self.right_b <= x

    def __gt__(self, x):
        return self.left_b > x

    def __ge__(self, x):
        return self.left_b >= x

    def __eq__(self, x):
        if isinstance(x, float):
            raise ValueError("not implemented eq")
        return super(_Bucket, self).__eq__(x)

    def __cmp__(self, x):
        raise ValueError("not implemented cmp")

    def add(self, loc):
        "Add new values in this bucket"
        self._bitmap.add(loc)

    def update(self, values):
        "Add new values in this bucket"
        self._bitmap.update(values)

    def remove(self, values):
        "Remove values from this bucket"
        self._bitmap = self._bitmap - bitmap.asbitmap(values)

    @staticmethod
    def find_bucket(alist, x):
        'Find the bucket whose bounds include x'
        i = bisect_right(alist, x)
        assert i > 0
        bkt = alist[i - 1]
        assert x >= bkt.left_b and x < bkt.right_b
        return i - 1, bkt


class _HistogramIndexImpl(object):
    def __init__(self, column, table, e_min, e_max, nb_bin):
        self._table = table
        self._column = column
        self.bins = nb_bin
        self.e_min = e_min
        self.e_max = e_max
        self._buckets = None
        self._init_histogram(e_min, e_max, nb_bin)
        #self._show_histogram()
        
    def _init_histogram(self, e_min, e_max, nb_bin):
        step = (e_max - e_min) * 1.0 / nb_bin
        left_b = float("-inf")
        self._buckets = []
        for i in range(nb_bin + 1):
            right_b = e_min + i * step
            bucket = _Bucket(left_b, right_b)
            self._buckets.append(bucket)
            left_b = right_b
        last_bkt = _Bucket(left_b, float("inf"))
        self._buckets.append(last_bkt)
    def _show_histogram(self):
        print("HISTOGRAM INDEX:")
        for bkt in self._buckets:
            print("bkt: ",dict(left=bkt.left_b, right=bkt.right_b))
        print("END HISTOGRAM INDEX")
    def reshape(self, min_, max_):
        "Change the bounds of the index if needed"
        pass  # to be defined...then implemented

    def _get_bin(self, val):
        return _Bucket.find_bucket(self._buckets, val)

    def update_histogram(self, created, updated, deleted):
        "Update the histogram index"
        updated_bm = bitmap.asbitmap(updated)
        if deleted or updated:
            deleted_bm = bitmap.asbitmap(deleted)
            to_remove = updated_bm | deleted_bm
            for bkt in self._buckets:
                bkt.remove(to_remove)
        if created or updated:
            created_bm = bitmap.asbitmap(created)
            to_add = updated_bm | created_bm
            for loc in to_add:
                # TODO: Extract the column and work on it
                # Should probably be done in cython
                x = self._table.at[loc, self._column]
                if np.isnan(x):
                    continue
                _, bin_ = self._get_bin(x)
                bin_.add(loc)

    def query(self, operator_, limit, approximate=False):  # blocking...
        """
        Return the list of rows matching the query.
        For example, returning all values less than 10 (< 10) would be
        `query(operator.__lt__, 10)`
        """
        pos, bkt = self._get_bin(limit)
        detail = bitmap([])
        if not approximate:
            for loc in bkt.values:
                x = self._table.at[loc, self._column]
                if operator_(x, limit):
                    detail.add(loc)

        if operator_ in (operator.lt, operator.le):
            values = functools.reduce(operator.or_,
                                          (b.values for b in self._buckets[:pos]), bitmap([]))
        else:
            values = functools.reduce(
                operator.or_, (b.values for b in self._buckets[pos + 1:]))
        return values | detail
        

EXAMPLE = [
    _Bucket(float("-inf"), -500.0),
    _Bucket(-500.0, -300),
    _Bucket(-300.0, 0.0),
    _Bucket(0.0, 50.0),
    _Bucket(50.0, 100.0),
    _Bucket(100, float("inf"))
]


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
            self._table = input_table
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
        else:
            # Many not always, or should the implementation decide?
            self._impl.reshape(bound_min, bound_max)
        deleted = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(step_size)
            steps += indices_len(deleted)
        created = None
        if input_slot.created.any():
            created = input_slot.created.next(step_size)
            steps += indices_len(created)
        updated = None
        if input_slot.updated.any():
            updated = input_slot.updated.next(step_size)
            steps += indices_len(updated)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        with input_slot.lock:
            input_table = input_slot.data()
        self._impl.update_histogram(created, updated, deleted)
        return self._return_run_step(
            self.next_state(input_slot), steps_run=steps)

    def _eval_to_ids(self, operator_, limit, input_ids=None):  # horribly slow
        input_slot = self.get_input_slot('table')
        table_ = input_slot.data()
        if input_ids is None:
            input_ids=indices_to_slice(table_.index)
        x = table_.loc[fix_loc(input_ids)][self.column].values
        mask_ = op(x, limit)
        arr = slice_to_arange(input_ids)
        return bitmap(arr[np.nonzero(mask_)[0]])  # maybe fancy indexing ...

    def query(self, operator_, limit):
        if self._impl:
            return self._impl.query(operator_, limit)  # we have an histogram so we query it
        # there are no histogram because init_threshold wasn't be reached yet
        # so we query the input table directly
        return self._eval_to_ids(operator_, limit)

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
