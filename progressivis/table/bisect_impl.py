from __future__ import absolute_import, division, print_function




from progressivis.core.utils import (slice_to_arange, slice_to_bitmap, 
                                     indices_len, fix_loc)
from progressivis.core.bitmap  import bitmap
import operator
import functools

from .mod_impl import ModuleImpl

import six

import numpy as np
import operator

_len = indices_len
_fix = fix_loc 

op_dict = {'>': operator.gt, '>=': operator.ge, '<': operator.lt,'<=': operator.le}

import logging
logger = logging.getLogger(__name__)

class _Selection(object):
    def __init__(self, values=None):
        self._values = bitmap([]) if values is None else values

    def add(self, values):
        self._values.update(values)

    def remove(self, values):
        self._values = self._values -bitmap(values)


    
class BisectImpl(ModuleImpl):
    def __init__(self, column, op, cache=None):
        super(BisectImpl,self).__init__()
        self._table = None
        self._prev_pivot = None
        self._column = column
        self._op = op
        if isinstance(op, str):
            self._op = op_dict[op]
        elif op not in op_dict.values():
            raise ValueError("Invalid operator {}".format(op))
        self.has_cache = False
        self.bins = None
        self.e_min = None
        self.e_max = None
        self.boundaries = None
        self._init_hist_cache(cache)
        
    def _eval_to_ids(self, pivot, input_ids):
        x = self._table.loc[_fix(input_ids), self._column][0].values
        mask_ = self._op(x, pivot)
        arr = slice_to_arange(input_ids)
        return bitmap(arr[np.nonzero(mask_)]) # maybe fancy indexing ...

    def _init_hist_cache(self,cache):
        if not cache:
            return
        if not isinstance(cache, tuple):
            raise ValueError("cache must be a tuple")
        if len(cache)!=3:
            raise ValueError("cache must be a 3 element tuple (e_min, e_max, nb_bin)")
        self.has_cache = True
        e_min, e_max, nb_bin = cache # i.e. estimated min, max. NB nb_bin means bins between e_min and e_max
        step = (e_max - e_min)*1.0 / nb_bin
        self.bins = [bitmap() for _ in range(nb_bin + 2)] # i.e [before min,...nb_bin, after_max]
        self.e_min = e_min
        self.e_max = e_max
        self.boundaries = [float("-inf")]+[e_min + i*step for i in range(nb_bin+1)]+[float("inf")]

    def _get_bin(self, val):
        pos = np.searchsorted(self.boundaries, val) -1
        return pos, self.bins[pos]

    def _update_hist_cache(self, created=None, updated=None, deleted=None):
        updated_bm = slice_to_bitmap(updated) if updated else bitmap([])
        if deleted or updated:
            deleted_bm = slice_to_bitmap(deleted) if deleted else bitmap([])
            to_remove = updated_bm | deleted_bm
            for i, b in enumerate(self.bins):
                self.bins[i] = b&to_remove
        if created or updated:
            created_bm = slice_to_bitmap(created) if created else bitmap([])
            to_add = updated_bm | created_bm
            for id in to_add:
                x = self._table.at[id, self._column]
                _, bin_ = self._get_bin(x)
                bin_.add(id)
            
    def _reconstruct_from_hist_cache(self, pivot):
        bi, _ = self._get_bin(pivot)
        result_bin = bitmap([])
        for id in self.bins[bi]:
            x = self._table.at[id, self._column]
            if self._op(x, pivot):
                result_bin.add(id)
        if self._op in (operator.lt, operator.le):
            values = functools.reduce(operator.or_, self.bins[:bi].append(result_bin))
        else:
            values = functools.reduce(operator.or_, self.bins[bi+1:].append(result_bin))
        self.result = _Selection(values)
        
    def resume(self, pivot, created=None, updated=None, deleted=None):
        def pivot_changed():
            return  self._prev_pivot is not None and pivot != self._prev_pivot
        if self.has_cache:
            self._update_hist_cache(created, updated, deleted)
        if pivot_changed():
            return self.reconstruct_from_hist_cache(pivot)
        if updated:
            self.result.remove(updated)
            res = self._eval_to_ids(pivot, updated)
            self.result.add(res)
        if created:
            res = self._eval_to_ids(pivot, created)
            self.result.add(res)
        if deleted:
            self.result.remove(deleted)
        
        
        
    def start(self, table, pivot, created=None, updated=None, deleted=None, cache=None):
        self._table = table
        self.result = _Selection()
        self._prev_pivot = None
        self._init_hist_cache(cache)
        self.is_started = True
        return self.resume(pivot, created, updated, deleted)
    
