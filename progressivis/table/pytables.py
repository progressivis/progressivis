"Wrapper for PyTables"
from __future__ import absolute_import, division, print_function

from collections import Iterable

import six
import numpy as np

from progressivis.utils.fast import indices_to_slice
from progressivis.utils.intdict import IntDict
from progressivis.core.utils import (integer_types, is_int)
from .table_sliced import BaseTable, TableSlicedView
from .dshape import dshape_from_pytable

from .column import BaseColumn


class ColumnPT(BaseColumn):
    "Column Wrapper for PyTables"
    def __init__(self, name, index, base, col_index):
        super(ColumnPT, self).__init__(name, index=index, base=base)
        self._col_index = col_index
    def __getitem__(self, key):
        if is_int(key):
            return self._base[key][self._col_index]
        vfunc = np.vectorize(lambda t: t[self._col_index])
        return vfunc(self._base[key])
    def __len__(self):
        return self._base.nrows

    @property
    def size(self):
        return len(self)

    def __setitem__(self, key, val):
        raise NotImplementedError

    @property
    def value(self):
        return self._base.col(self._name)

    def resize(self, newsize):
        raise NotImplementedError

    def shape(self):
        return (len(self),) + self.dshape

    def set_shape(self, shape):
        raise NotImplementedError

    @property
    def fillvalue(self):
        raise NotImplementedError

    @property
    def maxshape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        return self._base.coltypes[self._name]

    @property
    def dshape(self):
        return self._base.coldescrs[self._name].shape

    @property
    def chunks(self):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError


class IdColumnPT(BaseColumn):
    "IDColumn Wrapper for PyTables"
    INTERNAL_ID = '_ID'
    def __init__(self, pt):
        super(IdColumnPT, self).__init__(IdColumnPT.INTERNAL_ID)
        self._base = np.arange(pt.nrows)
        self._last_id = self.size - 1
        self._ids_dict = None
    def __getitem__(self, key):
        return self._base[key]

    def __len__(self):
        return len(self._base)

    @property
    def size(self):
        return len(self)

    def __setitem__(self, key, val):
        raise NotImplementedError

    @property
    def value(self):
        return self._base

    @property
    def values(self):
        return self.value

    def resize(self, newsize):
        raise NotImplementedError

    @property
    def shape(self):
        return (len(self),)

    def set_shape(self, shape):
        raise NotImplementedError

    @property
    def maxshape(self):
        raise NotImplementedError

    @property
    def fillvalue(self):
        raise NotImplementedError

    @property
    def dtype(self):
        return self._base.dtype

    @property
    def dshape(self):
        return self._base.coldescrs[self._name].shape

    @property
    def chunks(self):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def _init_ids_dict(self, end=None):
        if end is None:
            end = self._last_id
        valid_ids = self._base[:end]
        new_ids = range(len(valid_ids))
        self._ids_dict = IntDict(valid_ids, new_ids)
        #self.add_updated(new_ids)

    def _update_ids_dict(self, start=0, end=None):
        if self._ids_dict is None:
            self._init_ids_dict(end)
        else:
            new_ids = self[start:end]
            self._ids_dict.update(new_ids, range(start, end))
            #TODO fix
            #self.add_updated(new_ids)

    def id_to_index(self, loc, as_slice=True):
        if self._ids_dict is None:
            self._update_ids_dict()
        if isinstance(loc, np.ndarray) and loc.dtype == np.int64:
            ret = self._ids_dict.get_items(loc)
        elif isinstance(loc, integer_types):
            if loc < 0:
                loc = self._last_id+loc
            return self._ids_dict[loc]
        elif isinstance(loc, Iterable):
            try:
                count = len(loc)
                # pylint: disable=bare-except
            except:
                count = -1
            ret = np.fromiter(loc, dtype=np.int64, count=count)
            ret = self._ids_dict.get_items(ret)
        elif isinstance(loc, slice):
            ret = np.array(range(loc.start, loc.stop+1, loc.step or 1), dtype=np.int64)
            ret = self._ids_dict.get_items(ret)
        else:
            raise ValueError('id_to_index not implemented for id "%s"' % loc)
        return indices_to_slice(ret) if as_slice else ret

    def equals(self, other):
        "Test if columns are equal"
        if self is other:
            return True
        return np.all(self.values == other.values)


class _PTLoc(object):
    "Wrapper for PyTables"
    # pylint: disable=too-few-public-methods
    def __init__(self, this_table, as_loc=True):
        self._table = this_table
        self._as_loc = as_loc
    def __delitem__(self, key):
        if isinstance(key, tuple):
            raise ValueError('getitem not implemented for key "%s"' % key)
        if self._as_loc: # i.e loc mode
            slice_maybe = self._table.id_to_index(key)
        else: # i.e iloc mode
            slice_maybe = key
        self._table.drop(slice_maybe)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('getitem not implemented for key "%s"' % key)
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)
        #pylint: disable=protected-access
        #cols = self._table[col_key]
        if self._as_loc: # i.e loc mode
            slice_maybe = self._table.id_to_index(row_key)
        else: # i.e iloc mode
            slice_maybe = row_key
        if isinstance(slice_maybe, slice) and slice_maybe.step in (None, 1):
            return TableSlicedView(self._table, slice_maybe, col_key)
        return self._table.create_subtable(slice_maybe, col_key, self._as_loc)
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('getitem not implemented for key "%s"' % key)
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)
        if self._as_loc:
            slice_maybe = self._table.id_to_index(row_key)
        else: # i.e iloc mode
            slice_maybe = row_key
        #pylint: disable=protected-access
        return self._table.setitem_2d(slice_maybe, col_key, value)

class _PTAt(object):
    "Wrapper for PyTables"
    # pylint: disable=too-few-public-methods
    def __init__(self, this_table, as_loc=True):
        self._table = this_table
        self._as_loc = as_loc
    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError('At getitem not implemented for key "%s"' % key)
        row_key, col_key = key
        if not isinstance(row_key, integer_types):
            raise ValueError(
                'At getitem not implemented for row key "%s"' % row_key)
        if self._as_loc:
            if not isinstance(col_key, six.string_types):
                raise ValueError(
                    'At getitem not implemented for column key "%s"' % col_key)
            row_key = self._table.id_to_index(row_key)
        else:
            if not isinstance(col_key, integer_types):
                raise ValueError(
                    'At getitem not implemented for column key "%s"' % col_key)
        return self._table._column(col_key)[row_key]

    def __setitem__(self, key, value):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise ValueError('At getitem not implemented for key "%s"' % key)
        row_key, col_key = key
        if not isinstance(row_key, integer_types):
            raise ValueError(
                'At getitem not implemented for row key "%s"' % row_key)
        if self._as_loc:
            if not isinstance(col_key, six.string_types):
                raise ValueError(
                    'At getitem not implemented for column key "%s"' % col_key)
            row_key = self._table.id_to_index(row_key)
        else:
            if not isinstance(col_key, integer_types):
                raise ValueError(
                    'At getitem not implemented for column key "%s"' % col_key)
        self._table._column(col_key)[row_key] = value


class PyTableView(BaseTable):
    "Wrapper for PyTables"
    def __init__(self, pt):
        super(PyTableView, self).__init__()
        self._pt = pt
        self._ids = IdColumnPT(pt)
        self._columns = []
        self._columndict = {}
        for i, name in enumerate(pt.colnames):
            column = ColumnPT(name, self._ids, pt, i)
            self._columns.append(column)
            self._columndict[name] = i

    @property
    def nrow(self):
        return self._pt.nrows

    def __len__(self):
        return self.nrow

    @property
    def columns(self):
        return self._columns

    def resize(self, newsize, index=None):
        pass # TODO: implement!

    def __delitem__(self, key):
        pass # TODO: implement!

    def drop(self, index, locs=None):
        pass # TODO: implement!

    def __getitem__(self, key):
        pass # TODO: implement!

    def __setitem__(self, key, values):
        pass # TODO: implement!

    @property
    def name(self):
        return self._pt.name

    @property
    def dshape(self):
        return dshape_from_pytable(self._pt)

    @property
    def loc(self):
        return _PTLoc(self)

    @property
    def iloc(self):
        return _PTLoc(self, as_loc=False)

    @property
    def at(self):
        return _PTAt(self)

    @property
    def iat(self):
        return _PTAt(self, as_loc=False)
