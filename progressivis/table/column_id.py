from collections import Iterable
import numpy as np
import logging

from progressivis.core.utils import (integer_types, is_none_alike,
                                     norm_slice, is_iterable)
from progressivis.utils.fast import indices_to_slice
from progressivis.utils.intdict import IntDict
from progressivis.core.index_update import IndexUpdate
from progressivis.core.config import get_option
from progressivis.core.bitmap import bitmap
from progressivis.storage.range import RangeDataset, RangeError

from .loc import Loc
from .column import Column
from . import metadata
from .dshape import dshape_create, dshape_to_h5py

logger = logging.getLogger(__name__)


class IdColumn:
    ATTR_LAST_ID = 'LAST_ID'
    INTERNAL_ID = '_ID'
    ID_DSHAPE = dshape_create("int64")
    ID_DTYPE = dshape_to_h5py(ID_DSHAPE)
    CHUNK_SIZE = 64*1024
    INVALID_ID = -1

    def __init__(self, storagegroup=None):
        self._index = self
        self._is_identity = True
        self._changes = None
        self._cached_index = IdColumn
        self.value = bitmap()

    @property
    def index(self):
        return self

    @property
    def is_identity(self):
        return self._is_identity

    @property
    def last_id(self):
        return self._last_id

    @property
    def changes(self):
        return self._changes

    @changes.setter
    def changes(self, c):
        self._flush_cache()
        self._changes = c

    @property
    def size(self):
        return len(self.value)
    
    def info_contents(self):
        if self.is_identity:
            return '[IDENTITY]'
        rep = '['
        max_rows = get_option('display.max_rows')
        for i, rid in enumerate(self):
            if i == max_rows:
                rep += "..."
                break
            rep += ("%d " % (rid))
        rep += ']'
        return rep

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, key):
        if is_iterable(key):
            return self.value | bitmap.asbitmap(key)
        return self.value[key]
    
    def resize(self, newsize, indices=None):
        """
        Change the size if of the IDColumn.
        When the column grows, return the new identifiers allocated.
        """
        # pylint: disable=arguments-differ
        oldsize = len(self.value)
        if oldsize == newsize:
            assert indices is None or len(indices) == 0
            return None
        elif oldsize > newsize:
            pass # TODO ... warning ?
        else:
            incr = newsize - oldsize
            assert indices is None or len(indices) == incr
            if indices is None:
                indices = bitmap(range(oldsize, newsize))
            self.value |= indices
        return indices

    def _allocate(self, count, indices=None):
        return self.resize(count, indices)

    
    def __delitem__(self, index):
        self.value -= bitmap.asbitmap(index)

    def __contains__(self, loc):
        v = Loc.dispatch(loc)
        end = self.size
        ids = self._ids_dict
        if v == Loc.INT:
            if self._is_identity:
                return 0 < loc < end
            else:
                return loc in ids
        if v == Loc.SLICE:
            if self._is_identity:
                return loc.start >= 0 and (loc.end is None or loc.end == end)
            else:
                loc = range(*loc.index(end))
                v = Loc.ITERABLE
        elif v == Loc.BITMAP:
            if self._is_identity:
                inside = bitmap(range(0, end))
                return loc.difference_cardinality(inside) == 0
            else:
                v = Loc.ITERABLE
        if Loc.isiterable(v):
            if self._is_identity:
                for ind in loc:
                    if ind < 0 or ind >= end:
                        return False
            else:
                for ind in loc:
                    if ind not in ids:
                        return False
            return True
        else:
            raise ValueError('Unsupported data for "in" %s', loc)

    def id_to_index(self, loc, as_slice=True):
        #import pdb;pdb.set_trace()
        ret = bitmap.asbitmap(loc) or self.value
        return ret.to_slice_maybe() if as_slice else ret

    def remove_module(self, mid):
        # TODO
        pass

    def _normalize_locs(self, locs):
        return bitmap.asbitmap(locs)

    def nonfree(self):
        indices = self.dataset[:]
        mask = np.ones(len(indices), dtype=np.bool)
        mask[self.freelist()] = False
        return indices, mask

    def to_array(self):
        return np.array(self.value)

    # begin(Change management)
    def _flush_cache(self):
        self._cached_index = IdColumn  # hack

    def touch(self, index=None):
        if index is self._cached_index:
            return
        self._cached_index = index
        self.add_updated(self[index])

    def add_created(self, locs):
        if self._changes:
            locs = self._normalize_locs(locs)
            self._changes.add_created(locs)

    def add_updated(self, locs):
        if self._changes:
            locs = self._normalize_locs(locs)
            self._changes.add_updated(locs)

    def add_deleted(self, locs):
        if self._changes:
            locs = self._normalize_locs(locs)
            self._changes.add_deleted(locs)

    def compute_updates(self, start, now, mid=None, cleanup=True):
        if self._changes:
            self._flush_cache()
            updates = self._changes.compute_updates(start, now, mid,
                                                    cleanup=cleanup)
            if updates is None:
                try:  # EAFP
                    updates = IndexUpdate(created=bitmap(self.dataset[:]))
                except OverflowError:
                    # because rows could be created then removed in same step
                    ids = self.dataset[:]
                    updates = IndexUpdate(created=bitmap(ids[ids >= 0]))
            return updates
        return None

    def equals(self, other):
        if self is other:
            return True
        return np.all(self.values == other.values)

    def create_dataset(*args, **kwargs):
        pass
    def load_dataset(self, dshape, nrow, shape=None):
        pass
