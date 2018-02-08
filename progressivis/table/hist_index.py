from __future__ import absolute_import, division, print_function

from .nary import NAry
from . import Table
from . import TableSelectedView
from ..core.slot import SlotDescriptor
from .module import Module
import numpy as np
from bisect import bisect_left, bisect_right

from progressivis.core.utils import (slice_to_arange, slice_to_bitmap,
                                     indices_len, fix_loc)
from progressivis.core.bitmap import bitmap
import operator
import functools

from .mod_impl import ModuleImpl

import six

import numpy as np
import operator

_len = indices_len
_fix = fix_loc

op_dict = {
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le
}

import logging
logger = logging.getLogger(__name__)


class _Selection(object):
    def __init__(self, values=None):
        self._values = bitmap([]) if values is None else values

    def add(self, values):
        self._values.update(values)

    def remove(self, values):
        self._values = self._values - bitmap(values)


class _Bucket(object):
    def __init__(self, left_b, right_b, bm=None):
        self._left_b = left_b
        self._right_b = right_b
        if bm is None:
            self._bitmap = bitmap([])
        else:
            self._bitmap = bm

    @property
    def values(self):
        return self._bitmap

    def __str__(self):
        return "[{}, {})".format(self._left_b, self._right_b)

    def __repr__(self):
        return self.__str__()

    def __lt__(self, x):
        ret = self._right_b < x
        return self._right_b < x

    def __le__(self, x):
        return self._right_b <= x

    def __gt__(self, x):
        return self._left_b > x

    def __ge__(self, x):
        return self._left_b >= x

    def __eq__(self, x):
        if isinstance(x, float):
            raise ValueError("not implemented eq")
        return super(_Bucket, self).__eq__(x)

    def __cmp__(self, x):
        raise ValueError("not implemented cmp")

    def add(self, loc):
        self._bitmap.add(loc)

    def update(self, values):
        self._bitmap.update(values)

    def remove(self, values):
        self._bitmap = self._bitmap - bitmap(values)

    @staticmethod
    def find_bucket(a, x):
        'Find the bucket whose bounds include x'
        i = bisect_right(a, x)
        assert i > 0
        bkt = a[i - 1]
        assert x >= bkt._left_b and x < bkt._right_b
        return i - 1, bkt


class _HistogramIndexImpl(ModuleImpl):
    def __init__(self, table_, column, e_min, e_max, nb_bin):
        super(_HistogramIndexImpl, self).__init__()
        self._table = table_
        self._column = column
        self.bins = nb_bin
        self.e_min = e_min
        self.e_max = e_max
        self._buckets = None
        self._init_histogram(e_min, e_max, nb_bin)

    def _init_histogram(self, e_min, e_max, nb_bin):
        step = (e_max - e_min) * 1.0 / nb_bin
        left_b = float("-inf")
        for i in range(nb_bin + 1):
            right_b = e_min + i * step
            bucket = _Bucket(left_b, right_b)
            self._buckets.append(bucket)
            left_b = right_b
        last_bkt = _Bucket(left_b, float("inf"))
        self._buckets.append(last_bkt)

    def reshape(self, min_, max_):
        pass  # to be defined...the implemented

    def _get_bin(self, val):
        return _Bucket.find_bucket(self._buckets, val)  # TODO: to be improved

    def _update_histogram(self, created=None, updated=None, deleted=None):
        updated_bm = slice_to_bitmap(updated) if updated else bitmap([])
        if deleted or updated:
            deleted_bm = slice_to_bitmap(deleted) if deleted else bitmap([])
            to_remove = updated_bm | deleted_bm
            for bkt in self._buckets:
                bkt.remove(to_remove)
        if created or updated:
            created_bm = slice_to_bitmap(created) if created else bitmap([])
            to_add = updated_bm | created_bm
            for loc in to_add:
                x = self._table.at[loc, self._column]
                _, bin_ = self._get_bin(x)
                bin_.add(loc)

    def query(self, operator_, pivot, approximate=False):
        pos, bkt = self._get_bin(pivot)
        detail = bitmap([])
        if not approximate:
            for loc in bkt.values:
                x = self._table.at[loc, self._column]
                if operator_(x, pivot):
                    detail.add(id)
        if operator_ in (operator.lt, operator.le):
            values = functools.reduce(operator.or_,
                                      (b.values for b in self._buckets[:pos]))
        else:
            values = functools.reduce(
                operator.or_, (b.values for b in self._buckets[pos + 1:]))
        return values | detail


sample = [
    _Bucket(float("-inf"), -500.0),
    _Bucket(-500.0, -300),
    _Bucket(-300.0, 0.0),
    _Bucket(0.0, 50.0),
    _Bucket(50.0, 100.0),
    _Bucket(100, float("inf"))
]


class HistogramIndexMod(Module):
    """
    """
    parameters = [
        ('bins', np.dtype(int), 128),
        ('init_threshold', int, 100),
    ]

    def __init__(self, column, scheduler=None, **kwds):
        """
        """
        self._add_slots(kwds, 'input_descriptors', [
            SlotDescriptor('table', type=Table, required=True),
            SlotDescriptor('min', type=Table, required=True),
            SlotDescriptor('max', type=Table, required=True)
        ])
        super(HistogramIndexMod, self).__init__(scheduler=scheduler, **kwds)
        self.column = column
        self._impl = None  # will be created when the init_threshold is reached
        # so realistic initial values for min and max were available
    def get_bounds(self, min_slot, max_slot):
        min_slot.created.next()
        with min_slot.lock:
            min_df = min_slot.data()
            if len(min_df) == 0 and self._bounds is None:
                return None
            min_ = min_df.last(self.column)

        max_slot.created.next()
        with max_slot.lock:
            max_df = max_slot.data()
            if len(max_df) == 0 and self._bounds is None:
                return None
            max_ = max_df.last(self.column)
        return (min_, max_)

    def run_step(self, run_number, step_size, howlong):
        input_slot = self.get_input_slot('table')
        input_slot.update(run_number, self.id)
        steps = 0
        with input_slot.lock:
            input_table = input_slot.data()
        if len(input_table) < self.params.init_threshold:
            # there are not enough rows. it's not worth building an index yet
            return self._return_run_step(self.state_blocked, steps_run=0)
        min_slot = self.get_input_slot('min')
        min_slot.update(run_number, self.id)
        max_slot = self.get_input_slot('max')
        max_slot.update(run_number, self.id)
        bounds = self.get_bounds(min_slot, max_slot)
        if bounds is None:
            logger.debug('No bounds yet at run %d', run_number)
            return self._return_run_step(self.state_blocked, steps_run=0)
        bound_min, bound_max = bounds
        if self._impl is None:
            HistogramIndexImpl(self.column, input_table, bound_min, bound_max,
                               self.params.bins)
        else:
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
        with input_slot.lock:
            input_table = input_slot.data()
        status = self._impl._update_histogram(
            created=created, updated=updated, deleted=deleted)
        return self._return_run_step(
            self.next_state(input_slot), steps_run=steps)

    def _eval_to_ids(self, op, pivot, input_ids=slice(0, None, 1)):
        input_slot = self.get_input_slot('table')
        table_ = input_slot.data()
        x = table_.loc[_fix(input_ids), self.column][0].values
        mask_ = op(x, pivot)
        arr = slice_to_arange(input_ids)
        return bitmap(arr[np.nonzero(mask_)])  # maybe fancy indexing ...

    def query(self, op, pivot):
        if self._impl:
            self._impl.query(op, pivot)  # we have an histogram so we query it
        else:
            self._eval_to_ids(op, pivot)  # there are no histogram yet so
            # we query the input table directly
