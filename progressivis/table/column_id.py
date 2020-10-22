from collections import Iterable
import numpy as np
import logging

from progressivis.core.utils import integer_types, is_none_alike, norm_slice
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


class IdColumn(Column):
    ATTR_LAST_ID = 'LAST_ID'
    INTERNAL_ID = '_ID'
    ID_DSHAPE = dshape_create("int64")
    ID_DTYPE = dshape_to_h5py(ID_DSHAPE)
    CHUNK_SIZE = 64*1024
    INVALID_ID = -1

    def __init__(self, storagegroup=None):
        super(IdColumn, self).__init__(IdColumn.INTERNAL_ID, None,
                                       storagegroup=storagegroup)
        self._index = self
        self._is_identity = True
        self._ids_dict = None
        self._last_id = 0
        self._changes = None
        self._cached_index = IdColumn  # hack
        self._freelist = bitmap()

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

    def freelist(self):
        return bitmap(self._freelist)

    def has_freelist(self):
        return bool(self._freelist)

    def freelist_size(self):
        return len(self._freelist)

    def info_contents(self):
        if self.is_identity:
            return '[IDENTITY]'
        rep = '['
        max_rows = get_option('display.max_rows')
            
        for i, rid in enumerate(self):
            if i == max_rows:
                rep += "..."
                break
            rep += ("%d "%(rid))
        rep += ']'
        return rep

    def __len__(self):
        return self.dataset.size-len(self._freelist)

    def __iter__(self):
        if self._freelist:
            return map(lambda x : self[x],
                        bitmap(range(0, self.size))-self._freelist)
        return iter(self.value)

    def create_dataset(self, dshape=None, fillvalue=-1, shape=None, chunks=None):
        assert(fillvalue==-1)
        if dshape is None:
            dshape=IdColumn.ID_DSHAPE
        else:
            assert(dshape==IdColumn.ID_DSHAPE)
        assert(fillvalue==-1)
        if not self._is_identity:
            dataset = super(IdColumn, self).create_dataset(dshape,
                                                           fillvalue,
                                                           shape=shape,
                                                           chunks=self.CHUNK_SIZE)
            self._last_id = 0
            dataset.attrs[IdColumn.ATTR_LAST_ID] = self._last_id
            return dataset

        # we copy Column.create_dataset because we want to avoid allocating a real dataset until
        # the ids are not the same as the indices
        self._dshape = dshape
        self._last_id = 0
        dataset = RangeDataset(self.name, shape=shape)        
        dataset.attrs[IdColumn.ATTR_LAST_ID] = self._last_id
        dataset.attrs[metadata.ATTR_COLUMN] = True
        dataset.attrs[metadata.ATTR_VERSION] = metadata.VALUE_VERSION
        dataset.attrs[metadata.ATTR_DATASHAPE] = str(dshape)
        self.dataset = dataset
        return dataset

    def _really_create_dataset(self, indices=None):
        logger.info('# Creating a real index for %s', self.name)
        olddataset = self.dataset
        dataset = super(IdColumn, self).create_dataset(self._dshape, -1,
                                                       shape=None,
                                                       chunks=(self.CHUNK_SIZE,))
        dataset.resize(olddataset.size)
        for start in range(0, self.size-1, self.CHUNK_SIZE):
            chunk=np.arange(start,
                            min(self.size, start+self.CHUNK_SIZE),
                            dtype=np.int64)
            dataset[chunk] = chunk
        dataset.attrs[IdColumn.ATTR_LAST_ID] = self._last_id
        dataset.attrs[metadata.ATTR_COLUMN] = True
        dataset.attrs[metadata.ATTR_VERSION] = metadata.VALUE_VERSION
        dataset.attrs[metadata.ATTR_DATASHAPE] = str(self._dshape)
        self._is_identity = False
        valid_ids = olddataset[:] if indices is None else indices
        if indices is None:
            self._ids_dict = IntDict(valid_ids, valid_ids)
        else:
            self._ids_dict = IntDict(valid_ids, np.arange(self.size))

    def load_dataset(self, dshape, nrow, shape=None):
        if dshape is None:
            dshape=IdColumn.ID_DSHAPE
        else:
            assert(dshape==IdColumn.ID_DSHAPE)
        dataset = super(IdColumn, self).load_dataset(dshape, nrow, shape, is_id=True)
        if dataset is None:
            self._is_identity = True
            dataset = self.create_dataset(dshape, -1, shape=shape)
            self.resize(nrow)
        else:
            self._is_identity = False
            self._last_id = dataset.attrs[IdColumn.ATTR_LAST_ID]
        return dataset

    def __setitem__(self, index, val):
        raise RuntimeError('setitem invalid for IdColumn')

    def _allocate(self, count, indices=None):
        if self._is_identity:
            if indices is not None:
                assert len(indices) == count
                try:
                    self.dataset[self.size:self.size+count] = indices
                except RangeError:
                    indices = np.asarray(indices, dtype=np.int64)
                    if np.any(np.logical_or(indices < self.size, indices < 0)):
                        raise ValueError('Indices contain duplicates')
                    # reboot the IDColumn to use a standard dataset
                    self._really_create_dataset()
                    return self._allocate(count, indices) # recursive call since we morphed
            return self.resize(self.size+count)
        # standard code using dataset/hash table
        if indices is None:
            indices = np.arange(self._last_id, self._last_id+count, dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)
        if self._ids_dict is not None and self._ids_dict.contains_any(indices):
            raise ValueError('Indices contain duplicates')
        # self._last_id = max(self._last_id, int(np.max(indices)+1))
        # NB: _last_id is set in resize()
        off = 0
        if self._freelist:
            alloc = self._freelist.pop(count)
            for i in alloc:
                newid = indices[off]
                self.dataset[i] = newid # filling the hole
                self._last_id = max(self._last_id, newid+1)
                indices[off] = i
                off += 1
                self._update_ids_dict(i,i+1)
        if off < count: # there are no holes OR there are more creations than holes
            # resize sets _last_id to max(indeices)+1
            new_indices = self.resize(self.size+count-off, indices[off:])
            indices[off:] = new_indices
        return indices

    def resize(self, newsize, indices=None):
        """
        Change the size if of the IDColumn.
        When the column grows, return the new identifiers allocated.
        """
        # pylint: disable=arguments-differ
        oldsize = self.size
        if oldsize == newsize:
            assert (indices is None or len(indices)==0)
            return None
        elif oldsize > newsize:
            todelete = self[newsize:]
            try:  # EAFP
                newsize_bm = bitmap(todelete)
                newsize = self._delete_ids(newsize_bm)
            except OverflowError:
                newsize_ = todelete[todelete >= 0]
                newsize = self._delete_ids(newsize_)
            if newsize is not None:
                super(IdColumn, self).resize(newsize)
                self._flush_cache()
            return None
        else:  # oldsize < newsize
            incr = newsize - oldsize
            assert indices is None or len(indices) == incr
            self._flush_cache()
            if self._is_identity:
                newindices = np.arange(oldsize, newsize)
                # if the new indices are not the same
                # as expected, allocate the hashtable-based storage.
                if indices is not None \
                   and not np.array_equal(indices, newindices):
                    self._really_create_dataset()  # indices=indices)
                    return self.resize(newsize, indices)
                # indices is None or == newindices, super.resize works
                super(IdColumn, self).resize(newsize)
                indices = newindices
                self.add_created(indices)
                self._last_id += incr
                self.dataset.attrs[IdColumn.ATTR_LAST_ID] = self._last_id
                return indices
            # not _is_identity, code using full dataset/hash table
            if indices is None:
                last_id = self._last_id+incr
                indices = np.arange(self._last_id, last_id, dtype=np.int64)
            else:
                indices = np.asarray(indices, dtype=np.int64)
                if self._ids_dict is not None \
                   and self._ids_dict.contains_any(indices):
                    raise ValueError('Indices would contain duplicates')
                last_id = max(self._last_id, int(np.max(indices)+1))
            # TODO reuse free list
            super(IdColumn, self).resize(newsize)
            self.dataset[oldsize:] = indices
            self._update_ids_dict(oldsize, oldsize+incr, indices)
            indices[:] = np.arange(oldsize, oldsize+incr)
            self._last_id = last_id
            self.dataset.attrs[IdColumn.ATTR_LAST_ID] = self._last_id
            return indices

    def __delitem__(self, index):
        if not self._is_identity:
            index = self[index]
        self._delete_ids(index)

    def _update_ids_dict(self, start=0, end=None, locs=None):
        if self._is_identity:
            self.add_created(self[start:end] if locs is None else locs)
        elif self._ids_dict is None:
            self._really_create_dataset()
        else:
            new_locs = self[start:end] if locs is None else locs
            self._ids_dict.update(new_locs, range(start, end))
            self.add_created(new_locs)

    def _delete_ids(self, locs, index=None):
        end = int(self.size)
        if self._is_identity:
            # Check if locs are contiguous to the end,
            # we can handle that by shrinking the index
            if isinstance(locs, integer_types):
                if locs == end-1:
                    self.add_deleted([locs])
                    super(IdColumn, self).resize(locs)
                    return end-1
            elif isinstance(locs, slice):
                locs = norm_slice(locs)
                start, stop, step = locs.start, locs.stop, locs.step 
                if stop==end and step==1:
                    if start==stop:
                        return end
                    self.add_deleted(slice(start, stop, step))
                    super(IdColumn, self).resize(start)
                    return end-(start-stop)
                if start < 0 or stop > end or start > stop:
                    raise ValueError('Invalid locs')
                if start == stop:
                    return 0
            elif isinstance(locs, bitmap):
                if len(locs)==0:
                    return 0
                if locs==bitmap(range(end-len(locs), end)):
                    self.add_deleted(locs)
                    super(IdColumn, self).resize(locs.min())
                    return end-len(locs)
                if locs.max() >= end:
                    raise ValueError('Invalid locs')
            elif isinstance(locs, Iterable):
                locs = np.asarray(locs)  # turn iterable into array
            if isinstance(locs, np.ndarray):
                if len(locs)==0:
                    return end
                if np.all(locs == np.arange(end-len(locs), end)):
                    self.add_deleted(locs)
                    super(IdColumn, self).resize(int(locs[0]))
                    return end-len(locs)
                if np.any(np.logical_or(locs > end, locs < 0)):
                    raise ValueError('Invalid locs')
            # Not contiguous to the end, need to morph with ids
            self._really_create_dataset()
            # Fall through
        ids = self._ids_dict
        assert ids is not None
        # TODO check len(index)==len(locs)
        if index is None:
            index = self.id_to_index(locs, as_slice=False)
        self.add_deleted(locs)
        if isinstance(locs, integer_types):
            locs = [locs]
            index = [index]
        elif isinstance(locs, slice):
            locs = range(*locs.indices(end))
        if isinstance(index, np.ndarray):
            index = np.nditer(index)  # Beware, nditer flattens the array, which is ok here
        elif isinstance(index, slice):
            index = range(index.start, index.stop, index.step if index.step else 1)
        try:
            for loc,idx in zip(locs, index):
                idx = int(idx)
                if idx == IdColumn.INVALID_ID:
                    logger.error('Invalid index -1 for id[%d] to delete', loc)
                    continue
                try:
                    del ids[loc]
                    if idx < end:
                        self.dataset[idx] = IdColumn.INVALID_ID
                        self._freelist.add(idx)
                except KeyError:
                    logger.error('Tried to delete nonexistent id %d', loc)
        except TypeError:
            logger.error('Unrecognized locs(%s) or index(%s) types',
                         locs, index)
            
        # Shrink the dataset if possible
        end -= 1
        old_end = end
        while end >= 0:
            if self.dataset[end] >= 0:
                break
            self._freelist.remove(end)
            end -= 1
        if old_end != end:
            super(IdColumn, self).resize(end+1)
        return end+1

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
                return loc.start >= 0 and (loc.end==None or loc.end==end)
            else:
                loc = range(*loc.index(end))
                v = Loc.ITERABLE
        elif v == Loc.BITMAP:
            if self._is_identity:
                inside = bitmap(range(0, end))
                return loc.difference_cardinality(inside)==0
            else:
                v = Loc.ITERABLE
        if Loc.isiterable(v):
            if self._is_identity:
                for l in loc:
                    if l < 0 or l >= end:
                        return False
            else:
                for l in loc:
                    if not l in ids:
                        return False
            return True
        else:
            raise ValueError('Unsupported data for "in" %s', loc)

    def id_to_index(self, loc, as_slice=True):
        if self._is_identity:
            if isinstance(loc, slice): # slices are inclusive
                if loc.stop != None:
                    return slice(loc.start, loc.stop+1, loc.step)
            if is_none_alike(loc):
                loc=slice(0, self.size, 1)
            elif isinstance(loc, integer_types):
                if loc < 0:
                    loc += self._last_id
            return loc
        if self._ids_dict is None:
            self._update_ids_dict()
        if is_none_alike(loc): # return everything
            """
            # this cannot work
            # because after many creations/deletions
            # some indices may be >self.size
            # and self._freelist may be empty
            # because holes were filled by new creations
            ret = bitmap(range(0, self.size))
            if self._freelist:
                ret -= self._freelist
            if as_slice:
                ret = ret.to_slice_maybe()
            return ret
            """
            loc = self.to_array()
            ret = self._ids_dict.get_items(loc) # no loc.copy() is needed here 
        elif isinstance(loc, np.ndarray) and loc.dtype==np.int:
            # NB: ALWAYS pass a COPY here (and below) because get_items() provides the result INPLACE!!!
            ret = self._ids_dict.get_items(loc.copy()) 
        elif isinstance(loc, integer_types):
            if loc < 0:
                loc = self._last_id+loc
            return self._ids_dict[loc]
        elif isinstance(loc, Iterable):
            try:
                count = len(loc)
                # pylint: disable=bare-except
            except:
                count=-1
            ret = np.fromiter(loc, dtype=np.int64, count=count)
            ret = self._ids_dict.get_items(ret)
        elif isinstance(loc, slice):
            loc_start = 0 if loc.start is None else loc.start
            loc_stop = self.last_id if loc.stop is None else loc.stop+1
            ret = np.array(range(loc_start, loc_stop, loc.step or 1), dtype=np.int64)
            try: # EAFP
                ret = self._ids_dict.get_items(ret)
            except KeyError: # occurs when ret contains deleted items (-1)
                ret_del = np.nonzero(self.index.dataset[ret]<0)[0]
                ret = self._ids_dict.get_items(np.array(bitmap(ret)-bitmap(ret_del), dtype=np.int64))
                                 
        else:
            raise ValueError('id_to_index not implemented for id "%s"' % loc)
        return indices_to_slice(ret) if as_slice else ret

    def remove_module(self, mid):
        # TODO
        pass

    def _normalize_locs(self, locs):
        if locs is None:
            if bool(self._freelist):
                locs = iter(self)
            else:
                locs = iter(self.dataset)
        elif isinstance(locs, integer_types):
            locs = [locs]
        return bitmap(locs)

    def nonfree(self):
        indices = self.dataset[:]
        mask = np.ones(len(indices), dtype=np.bool)
        mask[self.freelist()] = False
        return indices, mask

    def to_array(self):
        if not self.has_freelist():
            return self.dataset[:]
        indices, mask = self.nonfree()
        return indices[mask]

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
