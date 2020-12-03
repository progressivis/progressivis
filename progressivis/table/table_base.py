"""Base class for Tables
"""
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict, Mapping, Iterable
import operator
import logging
import numpy as np
import weakref
from progressivis.core.index_update import IndexUpdate
from progressivis.core.utils import (integer_types, norm_slice, is_slice,
                                     slice_to_bitmap, is_int, is_str,
                                     all_int, all_string, is_iterable,
                                     all_string_or_int, all_bool,
                                     indices_len, remove_nan,
                                     is_none_alike, get_physical_base)
from progressivis.core.config import get_option
from progressivis.core.bitmap import bitmap
from .dshape import dshape_print, dshape_create

logger = logging.getLogger(__name__)


FAST = 1


class _BaseLoc(object):
    # pylint: disable=too-few-public-methods
    def __init__(self, this_table, as_loc=True):
        self._table = this_table
        self._as_loc = as_loc

    def parse_key(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('getitem not implemented for key "%s"' % key)
            index, col_key = key
        else:
            index, col_key = key, slice(None)
        locs = None
        if self._as_loc:  # i.e loc mode
            locs = index
            index = self._table._any_to_bitmap(index)
        return index, col_key, locs

    def parse_key_to_bitmap(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('getitem not implemented for key "%s"' % key)
            raw_index, col_key = key
        else:
            raw_index, col_key = key, slice(None)
        index = self._table._any_to_bitmap(raw_index)
        return index, col_key, raw_index


class _Loc(_BaseLoc):
    # pylint: disable=too-few-public-methods
    def __delitem__(self, key):
        index, col_key, raw_index = self.parse_key_to_bitmap(key)
        if not is_none_alike(col_key):
            raise ValueError('Cannot delete key "%s"' % key)
        self._table.drop(index, raw_index)

    def __getitem__(self, key):
        index, col_key, raw_index = self.parse_key_to_bitmap(key)
        if not (is_slice(raw_index) or index in self._table._index):
            diff_ = index - self._table._index
            raise KeyError(f"Not existing indices {diff_}")
        if isinstance(raw_index, integer_types):
            row = self._table.row(raw_index)
            if col_key != slice(None):
                return row[col_key]
            return row
        elif isinstance(index, Iterable):
            index = bitmap.asbitmap(index)
            btab = BaseTable(index=index)
            columns, columndict = self._table.make_projection(col_key, btab)
            btab._columns = columns
            btab._columndict = columndict
            btab._dshape = dshape_create(
            '{'
            + ','.join(["{}:{}".format(c.name, c.dshape)
                       for c in btab._columns])
            + '}')
            btab.observe(self._table)
            return btab
        raise ValueError('getitem not implemented for index "%s"', index)

    def __setitem__(self, key, value):
        index, col_key, _ = self.parse_key(key)
        return self._table.setitem_2d(index, col_key, value)


class _At(_BaseLoc):
    # pylint: disable=too-few-public-methods
    def parse_key(self, key):
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('getitem not implemented for key "%s"' % key)
            index, col_key = key
        else:
            raise KeyError(f"Invalid key {key}")
        if not is_int(index):
            raise KeyError(f"Invalid row key {index}")
        if not (is_str(col_key) or is_int(col_key)):
            raise KeyError(f"Invalid column key {col_key}")
        return index, col_key

    def __getitem__(self, key):
        index, col_key = self.parse_key(key)
        if index not in self._table._index:
            raise KeyError(f"Not existing indice {index}")
        return self._table[col_key][index]

    def __setitem__(self, key, value):
        index, col_key = self.parse_key(key)
        if not is_int(index):
            raise KeyError(f"Invalid row key {index}")
        if not  (is_str(col_key) or is_int(col_key)):
            raise ValueError('At setitem not implemented for column key "%s"' %
                             col_key)
        self._table[col_key][index] = value


class BaseTable(metaclass=ABCMeta):
    # pylint: disable=too-many-public-methods, too-many-instance-attributes
    """Base class for Tables.
    """
    def __init__(self, base=None, index=None, columns=None, columndict=None):
        self._base = base
        self._columns = [] if columns is None else columns
        self._columndict = OrderedDict() if columndict is None else columndict
        self._index = bitmap() if index is None else index
        self._loc = _Loc(self, True)
        self._at = _At(self, True)
        self._changes = None
        #self._is_identity = True
        self._cached_index = BaseTable # hack
        self._last_id = -1
        self._observed = None
        self._observers = []

    @property
    def loc(self):
        "Return a `locator` object for indexing using ids"
        return self._loc

    @property
    def at(self):
        # pylint: disable=invalid-name
        "Return an object for indexing values using ids"
        return self._at

    def __repr__(self):
        return str(self) + self.info_contents()

    def __str__(self):
        classname = self.__class__.__name__
        length = len(self)
        return u'%s("%s", dshape="%s")[%d]' % (classname,
                                               self.name,
                                               dshape_print(self.dshape),
                                               length)

    def observe(self, other):
        if self._observed:
            raise ValueError("Observed table already set")
        self._observed = other
        other._observers.append(weakref.ref(self))

    def get_original(self):
        if self._observed is None:
            return self
        return self._observed.get_original()

    def info_row(self, row, width):
        "Return a description for a row, used in `repr`"
        row_id = row if row in self._index else -1
        rep = "{0:{width}}|".format(row_id, width=width)
        for name in self.columns:
            col = self[name]
            v = str(col[row])
            if len(v) > width:
                if col.dshape == "string":
                    v = v[0:width-3]+'...'
                else:
                    v = v[0:width-1]+'.'
            rep += ("{0:>{width}}|".format(v, width=width))
        return rep

    def info_contents(self):
        "Return a description of the contents of this table"
        length = len(self)
        rep = ''
        max_rows = min(length, get_option('display.max_rows'))
        if max_rows == 0:
            return ''
        if max_rows < length:
            head = max_rows//2
            tail = length - max_rows//2
        else:
            head = length
            tail = None
        width = get_option('display.column_space')

        rep += ("\n{0:^{width}}|".format("Index", width=width))
        for name in self.columns:
            if len(name) > width:
                name = name[0:width]
            rep += ("{0:^{width}}|".format(name, width=width))

        for row in range(head):
            rep += "\n"
            rep += self.info_row(row, width)

        if tail:
            rep += ("\n...(%d)..." % length)
            for row in range(tail, length):
                rep += "\n"
                rep += self.info_row(row, width)
        return rep

    def index_to_mask(self):
        return np.array(((elt in self._index) for elt in range(self.last_id+1)))

    def index_to_array(self):
        return np.array(self._index, dtype='int32')

    def __iter__(self):
        return iter(self._columndict.keys())

    @property
    def size(self):
        "Return the size of this table, which is the number of rows"
        return self.nrow

    @property
    def is_identity(self):
        "Return True if the index is using the identity mapping"
        sl = self._index.to_slice_maybe()
        if not isinstance(sl, slice):
            return False
        return sl == slice(0, self.last_id+1, None)

    @property
    def last_id(self):
        "Return the last id of this table"
        #if not self._index:
        #    assert self._last_id == -1
        if self._index and self._last_id < self._index.max():
            self._last_id = self._index.max()
        return self._last_id

    def width(self, colnames=None):
        """Return the number of effective width (number of columns) of the table

        Since a column can be multidimensional, the effective width of a table
        is the sum of the effective width of each of its columns.

        Parameters
        ----------
        colnames : list or `None`
            The optional list of columns to use for counting, or all the
            columns when not specified or `None`.
        """
        columns = self._columns if colnames is None else [self[name]
                                                          for name in colnames]
        width = 0
        for col in columns:
            width += col.shape[1] if len(col.shape) > 1 else 1
        return width

    @property
    def shape(self):
        "Return the shape of this table as if it were a numpy array"
        return self.size, self.width()

    def to_json(self, **kwds):
        "Return a dictionary describing the contents of this columns."
        return self.to_dict(**kwds)
    def make_col_view(self, col, index):
        from .column_selected import ColumnSelectedView
        return ColumnSelectedView(base=col, index=index, name=col.name)
    def make_projection(self, cols, index):
        dict_ = self._make_columndict_projection(cols)
        columns = [self.make_col_view(c, index) for (i, c) in enumerate(self._columns) if i in  dict_.values()]
        columndict = OrderedDict(zip(dict_.keys(), range(len(dict_))))
        return columns, columndict

    def _make_columndict_projection(self, cols):
        if is_none_alike(cols):
            return self._columndict
        if isinstance(cols, slice):
            assert is_int(cols.start) # for the moment ...
            nsl = norm_slice(cols, stop=len(self._columndict))
            return OrderedDict({k:v for (i, (k,v)) in enumerate(self._columndict.items())
                    if i in range(*nsl.indices(nsl.stop))})
        if is_int(cols) or is_str(cols):
            cols = [cols]
        if is_iterable(cols):
            if all_int(cols):
                return OrderedDict({k:v for (i, (k,v)) in enumerate(self._columndict.items())
                                    if i in cols})
            if all_string(cols):
                return OrderedDict({k:v for (k,v) in self._columndict.items()
                    if k in cols})
        raise ValueError(f"Invalid column projection {cols}")

    def to_dict(self, orient='dict', columns=None):
        # pylint: disable=too-many-branches
        """
        Return a dictionary describing the contents of this columns.

        Parameters
        ----------
        orient : {'dict', 'list', 'split', 'rows', 'record', 'index'}
            TODO
        columns : list or `None`
            TODO
        """
        if columns is None:
            columns = self.columns
        if orient == 'dict':
            ret = OrderedDict()
            for name in columns:
                col = self[name]
                ret[name] = {
                    int(k): v
                    for (k, v) in dict(zip(self.index,
                                           col.tolist())).items()
                }  # because a custom JSONEncoder cannot fix it
            return ret
        if orient == 'list':
            ret = OrderedDict()
            for name in columns:
                col = self[name]
                ret[name] = col.tolist()
            return ret
        if orient == 'split':
            ret = {'index': list(self.index),
                   'columns': columns}
            data = []
            cols = [self[c] for c in columns]
            for i in self.index:
                line = []
                for col in cols:
                    line.append(get_physical_base(col).loc[i])
                data.append(line)
            ret['data'] = data
            return ret
        if orient == 'datatable':
            # not a pandas compliant mode but useful for JS DataTable
            ret = []
            for i in self.index:
                line = [i]
                for name in columns:
                    col = self[name]
                    line.append(remove_nan(get_physical_base(col).loc[i]))
                ret.append(line)
            return ret
        if orient in ('rows', 'records'):
            ret = []
            for i in self.index:
                line = OrderedDict()
                for name in columns:
                    col = self[name]
                    line[name] = get_physical_base(col).loc[i]
                ret.append(line)
            return ret
        if orient == 'index':
            ret = OrderedDict()
            for id_ in self.index:
                line = {}
                for name in columns:
                    col = self[name]
                    line[name] = col.loc[id_]
                ret[int(id_)] = line
            return ret
        raise ValueError(f"to_dict({orient}) not implemented")

    def to_csv(self, filename, columns=None, sep=','):  # TODO: to be improved
        if columns is None:
            columns = self.columns
        with open(filename, 'wb') as f:
            for i in self.index:
                row = []
                for name in columns:
                    col = self[name]
                    row.append(str(remove_nan(get_physical_base(col).loc[i])))
                row = sep.join(row)
                f.write(row.encode('utf-8'))
                f.write(b'\n')

    def column_offsets(self, columns, shapes=None):
        '''Return the offsets of each column considering columns can have
        multiple dimensions
        '''
        if shapes is None:
            shapes = [self[c].shape for c in columns]
        offsets = [0]
        dim2 = 0
        for shape in shapes:
            dims = len(shape)
            if dims > 2:
                raise ValueError('Cannot convert table to numpy array because'
                                 'of shape %s', shape)
            dim2 += dims
            offsets.append(dim2)
        return offsets

    @property
    def columns(self):
        "Return the list of column names in this table"
        return list(self._columndict.keys())

    def _column(self, name):
        if isinstance(name, integer_types):
            return self._columns[name]
        return self._columns[self._columndict[name]]

    def column_index(self, name):
        "Return the index of the specified column in this table"
        if isinstance(name, integer_types):
            return name
        return self._columndict[name]

    def index_to_id(self, ix):
        """Return the ids of the specified indices
        NB: useless for this implementation. kept for compat.
        Parameters
        ----------
        ix: the specification of an index or a list of indices
            The list can be specified with multiple formats: integer, list,
            numpy array, Iterable, or slice.  A similar format is return,
            except that slices and Iterables may return expanded as lists or
            arrays.
        """
        if is_int(ix):
            return ix
        locs =  self._any_to_bitmap(ix)
        assert locs in self._index
        return locs

    def id_to_index(self, loc, as_slice=True):
        # to be reimplemented with LRU-dict+pyroaring
        """Return the indices of the specified id or ids
        NB: useless for this implementation. kept for compat.
        Parameters
        ----------
        loc : an id or list of ids
            The format can be: integer, list, numpy array, Iterable, or slice.
            Note that a bitmap is an list, and array, and a bitmap are all
            Iterables but are managed in an efficient way.
        as_slice : boolean
            If True, try to convert the result into a slice if possible and
            not too expensive.
        """
        return self.index_to_id(loc)

    @property
    def index(self):
        "Return the object in change of indexing this table"
        return self._index

    @index.setter
    def index(self, indices):
        "Modify the object in change of indexing this table"
        if self._observed is None:
            raise ValueError("Cannot modify the index of a physical table this way. Use append instead")
        indices = self._any_to_bitmap(indices)
        if indices not in self._observed.index:
            raise ValueError(f"Not existing indices {indices-self._observed.index}")
        created_ = indices - self._index
        if created_:
            self.add_created(created_)
        deleted_ = self._index - indices
        if deleted_:
            self.add_deleted(deleted_)
        self._index = indices
        return self._index

    @property
    def ncol(self):
        "Return the number of columns (same as `len(table.columns()`)"
        return len(self._columns)

    @property
    def nrow(self):
        "Return the number of rows (same as `len(table)`)"
        return len(self._index)

    def __len__(self):
        return self.nrow

    def _slice_to_bitmap(self, sl, fix_loc=True, existing_only=True):
        stop = sl.stop or self.last_id
        nsl = norm_slice(sl, fix_loc, stop=stop)
        ret = bitmap(nsl)
        if existing_only:
            ret &= self._index
        return ret

    def _any_to_bitmap(self, locs, copy=True, fix_loc=True, existing_only=True):
        if isinstance(locs, bitmap):
            return locs[:] if copy else locs
        if isinstance(locs, integer_types):
            return bitmap([locs])
        if isinstance(locs, Iterable):
            if all_bool(locs):
                return bitmap(np.nonzero(locs))
            else:
                return bitmap(locs)
        if isinstance(locs, slice):
            return self._slice_to_bitmap(locs, fix_loc, existing_only)
        raise KeyError(f"Invalid type {type(locs)} for key {locs}")


    def __delitem__(self, key):
        bm = self._any_to_bitmap(key)
        if isinstance(key, Iterable):
            assert bm in self._index
        self._index -= bm
        self.add_deleted(bm)

    def drop(self, index, raw_index=None):
        "index is useless by now"
        if raw_index is None:
            raw_index = index
        if is_int(raw_index):
            #self.__delitem__(raw_index)
            self._index.remove(raw_index)
        else:
            #self.__delitem__(index)
            self._index -= index
        self.add_deleted(index)

    def _resize_rows(self, newsize, index=None):
        #self._ids.resize(newsize, index)
        if index is not None:
            index = self._any_to_bitmap(index)
            if index and index.min() > self.last_id:
                self._index |= index
            else:
                assert index in self._index
                self._index = index
        else:
            #assert self._is_identity
            if newsize >= self.last_id+1:
                self._index |= bitmap(range(self.last_id+1, newsize))
            else:
                self._index &=  bitmap(range(0, newsize))


    @property
    def name(self):
        "Return the name of this table"
        pass

    @property
    def dshape(self):
        "Return the datashape of this table"
        return self._dshape

    @property
    def base(self):
        "Return the base table for views, or None if the table is not a view"
        return self._base

    @property
    def changes(self):
        "Return the TableChange manager associated with this table or None"
        #if not self._index:
        #    return None
        return self._changes

    @changes.setter
    def changes(self, tablechange):
        "Set the TableChange manager, or unset with None"
        #if not self._index:
        #    raise RuntimeError('Table has no index')
        self._changes = tablechange

    def compute_updates(self, start, now, mid=None, cleanup=True):
        """Compute the updates (delta) that happened to this table since the last call.

        Parameters
        ----------
        start: integer
            Start is interpreted as a virtual time for `last time`
        now: integer
            Start is interpreted as a virtual time for `now`
        mid: hashable object
            An identifier for the object that will ask for updates,
            usually the name of a slot.
        Returns
        -------
        updates: None or an IndexUpdate structure which describes the list
             of rows created, updated, and deleted.
        """
        if not self._index:
            return None
        if self._changes:
            updates = self._changes.compute_updates(start, now, mid,
                                                    cleanup=cleanup)
            if updates is None:
                updates = IndexUpdate(created=bitmap(self._index)) # pass an index copy ...
            return updates
        return None


    def __getitem__(self, key):
        # hack, use t[['a', 'b'], 1] to get a list instead of a TableView
        fast = False
        if isinstance(key, tuple):
            key = key[0] # i.e. columns
            fast = True
        if isinstance(key, (str, integer_types)):
            return self._column(key)
        elif isinstance(key, Iterable):
            if fast:
                return (self._column(c) for c in key)
            if all_bool(key):
                return self.iloc[key]
        elif isinstance(key, slice):
            if fast:
                indices = self._col_slice_to_indices(key)
                return (self._column(c) for c in range(*indices))
        raise ValueError('getitem not implemented for key "%s"' % key)

    def row(self, loc):
        "Return a Row object wrapping the loc"
        return self.last(loc)

    def iterrows(self):
        "Return an iterator returning rows and their ids"
        return map(self.row, iter(self._index))

    def last(self, key=None):
        "Return the last row"
        length = len(self)
        if length == 0:
            return None
        if key is None or isinstance(key, integer_types):
            from .row import Row
            return Row(self, key)
        if isinstance(key, str):
            return self._column(key)[self.last_id]
        if all_string_or_int(key):
            index = self.last_id
            return (self._column(c)[index] for c in key)
        raise ValueError('last not implemented for key "%s"' % key)

    def setitem_2d(self, rowkey, colkey, values):
        if isinstance(colkey, (str, integer_types)):
            self._setitem_key(colkey, rowkey, values)
        elif isinstance(colkey, Iterable):
            self._setitem_iterable(colkey, rowkey, values)
        elif isinstance(colkey, slice):
            self._setitem_slice(colkey, rowkey, values)
        else:
            raise ValueError("Unhandled key %s", colkey)

    def __setitem__(self, colkey, values):
        if isinstance(colkey, tuple):
            raise ValueError("Adding new columns ({}) via __setitem__"
                             " not implemented".format(colkey))
        if isinstance(colkey, (str, integer_types)):
            # NB: on Pandas, only strings are accepted!
            self._setitem_key(colkey, None, values)
        elif isinstance(colkey, Iterable):
            if not all_string_or_int(colkey):
                raise ValueError("setitem not implemented for %s key" % colkey)
            self._setitem_iterable(colkey, None, values)
        elif isinstance(colkey, slice):
            self._setitem_slice(colkey, None, values)
        else:
            raise ValueError("Unhandled key %s", colkey)

    def _setitem_key(self, colkey, rowkey, values):
        if is_none_alike(rowkey) and len(values) != len(self):
            raise ValueError("Length of values (%d) different "
                             "than length of table (%d)" % (
                                 len(values), len(self)))
        column = self._column(colkey)
        if is_none_alike(rowkey):
            column[:] = values
        else:
            column[rowkey] = values

    def _setitem_iterable(self, colkey, rowkey, values):
        # pylint: disable=too-many-branches
        colnames = list(colkey)
        len_colnames = len(colnames)
        if not isinstance(values, Iterable):
            values = np.repeat(values, len_colnames)
        if isinstance(values, Mapping):
            for (k, v) in values.items():
                column = self._column(k)
                if is_none_alike(rowkey):
                    column[:] = v
                else:
                    column[rowkey] = v
        elif hasattr(values, 'shape'):
            shape = values.shape
            if len(shape) > 1 and shape[1] != self.width(colnames):
                # and not isinstance(values, BaseTable):
                raise ValueError('Shape [1] (width)) of columns and '
                                 'value shape do not match')

            if rowkey is None:
                rowkey = slice(None, None)
            for i, colname in enumerate(colnames):
                column = self._column(colname)
                if len(column.shape) > 1:
                    wid = column.shape[1]
                    column[rowkey, 0:wid] = values[i:i+wid]
                else:  # i.e. len(column.shape) == 1
                    if isinstance(values, BaseTable):
                        column[rowkey] = values[i]
                    elif len(shape) == 1:  # values is a row
                        column[rowkey] = values[i]
                    else:
                        column[rowkey] = values[:, i]
        else:
            for i, colname, v in zip(range(len_colnames), colnames, values):
                column = self._column(colname)
                if is_none_alike(rowkey):
                    column[:] = v
                else:
                    column[rowkey] = values[i]

    def _col_slice_to_indices(self, colkey):
        if isinstance(colkey.start, str):
            start = self.column_index(colkey.start)
            end = self.column_index(colkey.stop)
            colkey = slice(start, end+1, colkey.step)
        return range(*colkey.indices(self.ncol))

    def _setitem_slice(self, colkey, rowkey, values):
        indices = self._col_slice_to_indices(colkey)
        self._setitem_iterable(indices, rowkey, values)

    def to_array(self, locs=None, columns=None, returns_indices=False):
        """Convert this table to a numpy array

        Parameters
        ----------
        locs: a list of ids or None
            The rows to extract.  Locs can be specified with multiple formats:
            integer, list, numpy array, Iterable, or slice.
        columns: a list or None
            the columns to extract
        """
        if columns is None:
            columns = self.columns

        shapes = [self[c].shape for c in columns]
        offsets = self.column_offsets(columns, shapes)
        dtypes = [self[c].dtype for c in columns]
        dtype = np.find_common_type(dtypes, [])
        indices = None
        # TODO split the copy in chunks
        if locs is None:
            indices = self._index
        else:
            indices = self._any_to_bitmap(locs)
        arr = np.empty((indices_len(indices), offsets[-1]), dtype=dtype)
        for i, column in enumerate(columns):
            col = self._column(column)
            shape = shapes[i]
            if len(shape) == 1:
                col.read_direct(arr, indices,
                                dest_sel=np.s_[:, offsets[i]])
            else:
                col.read_direct(arr, indices,
                                dest_sel=np.s_[:, offsets[i]:offsets[i+1]])
        if returns_indices:
            return indices, arr
        return arr

    def unary(self, op, **kwargs):
        axis = kwargs.get('axis', 0)
        # get() is cheaper than pop(), it avoids to update unused kwargs
        keepdims = kwargs.get('keepdims', False)
        # ignore other kwargs, maybe raise error in the future
        res = OrderedDict()
        for col in self._columns:
            value = op(col.values, axis=axis, keepdims=keepdims)
            res[col.name] = value
        return res

    def raw_unary(self, op, **kwargs):
        res = OrderedDict()
        for col in self._columns:
            value = op(col.values, **kwargs)
            res[col.name] = value
        return res

    def binary(self, op, other, **kwargs):
        axis = kwargs.pop('axis', 0)
        assert axis == 0
        res = OrderedDict()
        isscalar = (np.isscalar(other) or isinstance(other, np.ndarray))
        for col in self._columns:
            name = col.name
            if isscalar:
                value = op(col, other)
            else:
                value = op(col, other[name])
            res[name] = value
        return res

    def __abs__(self, **kwargs):
        return self.unary(np.abs, **kwargs)

    def __add__(self, other):
        return self.binary(operator.add, other)

    def __radd__(self, other):
        return other.binary(operator.add, self)

    def __and__(self, other):
        return self.binary(operator.and_, other)

    def __rand__(self, other):
        return other.binary(operator.and_, self)

    # def __div__(self, other):
    #     return self.binary(operator.div, other)

    # def __rdiv__(self, other):
    #     return other.binary(operator.div, self)

    def __eq__(self, other):
        return self.binary(operator.eq, other)

    def __gt__(self, other):
        return self.binary(operator.gt, other)

    def __ge__(self, other):
        return self.binary(operator.ge, other)

    def __invert__(self):
        return self.unary(np.invert)

    def __lshift__(self, other):
        return self.binary(operator.lshift, other)

    def __rlshift__(self, other):
        return other.binary(operator.lshift, self)

    def __lt__(self, other):
        return self.binary(operator.lt, other)

    def __le__(self, other):
        return self.binary(operator.le, other)

    def __mod__(self, other):
        return self.binary(operator.mod, other)

    def __rmod__(self, other):
        return other.binary(operator.mod, self)

    def __mul__(self, other):
        return self.binary(operator.mul, other)

    def __rmul__(self, other):
        return other.binary(operator.mul, self)

    def __ne__(self, other):
        return self.binary(operator.ne, other)

    def __neg__(self):
        return self.unary(np.neg)

    def __or__(self, other):
        return self.binary(operator.or_, other)

    def __pos__(self):
        return self

    def __ror__(self, other):
        return other.binary(operator.or_, self)

    def __pow__(self, other):
        return self.binary(operator.pow, other)

    def __rpow__(self, other):
        return other.binary(operator.pow, self)

    def __rshift__(self, other):
        return self.binary(operator.rshift, other)

    def __rrshift__(self, other):
        return other.binary(operator.rshift, self)

    def __sub__(self, other):
        return self.binary(operator.sub, other)

    def __rsub__(self, other):
        return other.binary(operator.sub, self)

    def __truediv__(self, other):
        return self.binary(operator.truediv, other)

    def __rtruediv__(self, other):
        return other.binary(operator.truediv, self)

    def __floordiv__(self, other):
        return self.binary(operator.floordiv, other)

    def __rfloordiv__(self, other):
        return other.binary(operator.floordiv, self)

    def __xor__(self, other):
        return self.binary(operator.xor, other)

    def __rxor__(self, other):
        return other.binary(operator.xor, self)

    def any(self, **kwargs):
        return self.unary(np.any, **kwargs)

    def all(self, **kwargs):
        return self.unary(np.all, **kwargs)

    def min(self, **kwargs):
        return self.unary(np.min, **kwargs)

    def max(self, **kwargs):
        return self.unary(np.max, **kwargs)

    def var(self, **kwargs):
        return self.raw_unary(np.var, **kwargs)

    def argmin(self, **kwargs):
        argmin_ = self.raw_unary(np.argmin, **kwargs)
        if self.is_identity:
            return argmin_
        index_array= self.index_to_array()
        for k, v in argmin_.items():
             argmin_[k] = index_array[v]
        return argmin_

    def argmax(self, **kwargs):
        argmax_ = self.raw_unary(np.argmax, **kwargs)
        if self.is_identity:
            return argmax_
        index_array= self.index_to_array()
        for k, v in argmax_.items():
             argmax_[k] = index_array[v]
        return argmax_

    def idxmin(self, **kwargs):
        res = self.argmin(**kwargs)
        for c, ix in res.items():
            res[c] = self.index_to_id(ix)
        return res

    def idxmax(self, **kwargs):
        res = self.argmax(**kwargs)
        for c, ix in res.items():
            res[c] = self.index_to_id(ix)
        return res




    def remove_module(self, mid):
        # TODO
        pass

    def _normalize_locs(self, locs):
        return self._any_to_bitmap(locs)
        """if locs is None:
            if bool(self._freelist):
                locs = iter(self)
            else:
                locs = iter(self.dataset)
        elif isinstance(locs, integer_types):
            locs = [locs]
        return bitmap(locs)"""

    def __old_nonfree(self):
        indices = self.dataset[:]
        mask = np.ones(len(indices), dtype=np.bool)
        mask[self.freelist()] = False
        return indices, mask

    def ___old_idcolumn_to_array(self):
        import pdb;pdb.set_trace()
        if not self.has_freelist():
            return self.dataset[:]
        indices, mask = self.nonfree()
        return indices[mask]

    # begin(Change management)
    def _flush_cache(self):
        self._cached_index = BaseTable

    def touch(self, index=None):
        if index is self._cached_index:
            return
        self._cached_index = index
        self.add_updated(index)

    def notify_observers(self, mode, locs):
        garbage = []
        for wr in self._observers:
            obs = wr()
            if obs is None:
                garbage.append(wr)
            else:
                obs.notification(mode, locs)
        for elt in garbage:
            self._observers.remove(elt)

    def add_created(self, locs):
        # self.notify_observers('created', locs)
        if self._changes:
            locs = self._normalize_locs(locs)
            self._changes.add_created(locs)

    def add_updated(self, locs):
        self.notify_observers('updated', locs)
        if self._changes:
            locs = self._normalize_locs(locs)
            self._changes.add_updated(locs)

    def add_deleted(self, locs):
        self.notify_observers('deleted', locs)
        if self._changes:
            locs = self._normalize_locs(locs)
            self._changes.add_deleted(locs)

    def notification(self, mode, locs):
        #if mode == 'created':
        #    pass
        locs = self._normalize_locs(locs)
        if mode == 'updated':
            self.add_updated(locs & self._index)
        elif mode == 'deleted':
            self.add_deleted(locs & self._index)
            self._index -= locs
        else:
            raise ValueError(f"Unknown mode {mode}")

    def compute_updates(self, start, now, mid=None, cleanup=True):
        if self._changes:
            self._flush_cache()
            updates = self._changes.compute_updates(start, now, mid,
                                                    cleanup=cleanup)
            if updates is None:
                updates = IndexUpdate(created=bitmap(self._index))
            return updates
        return None

    def equals(self, other):
        if self is other:
            return True
        return np.all(self.values == other.values)
