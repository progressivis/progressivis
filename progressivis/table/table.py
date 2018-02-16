# -*- coding: utf-8 -*-
"""
Main Table class
"""
from __future__ import absolute_import, division, print_function

from collections import OrderedDict, Mapping
import logging

import numpy as np
import pandas as pd
import numexpr as ne

import six

from progressivis.core.utils import (integer_types, get_random_name,
                                     all_int, are_instances, gen_columns)
from progressivis.core.fast import indices_to_slice
from progressivis.core.storage import Group
from .dshape import (dshape_create, dshape_table_check, dshape_fields,
                     dshape_to_shape, dshape_extract, dshape_compatible,
                     dshape_from_dtype)
from . import metadata
from .table_base import BaseTable
from .column import Column
from .column_id import IdColumn

if six.PY2:
    from itertools import imap
else:
    imap = map

logger = logging.getLogger(__name__)

__all__ = ["Table"]


class Table(BaseTable):
    """Create a Table data structure, made of a collection of columns.

    A Table is similar to Python Pandas or R DataFrame, but
    column-based and supporting fast addition of items.

    Args:
        name : string
            The name of the table
        data : optional container
            Data that will be appended to the table. It can be of multiple
            types.
            Table) another table is used to fill-up this table
            DataFrame) a Pandas DataFrame is copied to this table
            ndarray) a numpy array is copied. The dshape should be provided
        dshape : data shape such as `{'a': int32, 'b': float64, 'c': string}`
            The column names and types as specified by the ``datashape`` library.
        fillvalues : the default values of the columns specified as a dictionary
            Each column is created with a default ``fillvalue``. This parameter can
            specify the fillvalue of each column with 3 formats:
            a single value) which will be used by all the column creations
            a dictionary) associating a column name to a value
            the '*' entry in a dictionary) for defaulting the fillvalue when not specified.
        storagegroup : a factory used to create the columns
            When not specified or `None` the default ``storage`` is used.
            Otherwise, a ``storagegroup`` is specified in Group.
        chunks : the specification of the chunking of columns when the storagegroup supports it
            Like the ``fillvalue`` argument, it can be one value or a dict.
        create : forces the creation of the column
            The the storagegroup allows persistence, a table with the same name may exist
            in the storagegroup. With ``create=False``, the previous value is loaded, whereas
            with ``create=True`` it is replaced.
        indices : the indices of the rows appended when data is specified, in case the
            table contents has to be joined with another table.

    Example:
        >>> from progressivis.table import Table
        >>> t = Table('my-table', dshape='{a: int32, b: float32, c: bool}')
        >>> len(t)
        0
        >>> t.columns()
        ['a', 'b', 'c']

    """
    def __init__(self,
                 name,
                 data=None,
                 dshape=None,
                 fillvalues=None,
                 storagegroup=None,
                 chunks=None,
                 create=None, indices=None):
        # pylint: disable=too-many-arguments, too-many-branches
        super(Table, self).__init__()
        if not (fillvalues is None or isinstance(fillvalues, Mapping)):
            raise ValueError('Invalid fillvalues (%s) should be None or a dictionary'%fillvalues)
        if not (chunks is None or isinstance(chunks, (integer_types, Mapping))):
            raise ValueError('Invalid chunks (%s) should be None or a dictionary'%chunks)
        if data is not None:
            if create is not None:
                logger.warning('creating a Table with data and create=False')
            create = True

        self._chunks = chunks
        #self._nrow = 0
        self._name = get_random_name('table_') if name is None else name
        # TODO: attach all randomly named tables to a dedicated, common parent node
        if not (storagegroup is None or isinstance(storagegroup, Group)):
            raise ValueError('Invalid storagegroup (%s) should be None or a Group'%storagegroup)
        if storagegroup is None:
            storagegroup = Group.default()
        if storagegroup is None:
            raise RuntimeError('Cannot get a valid default storage Group')
        self._storagegroup = storagegroup
        if dshape is None:
            if data is None:
                self._dshape = None
            else:
                data = self.parse_data(data)
                self._dshape = dshape_extract(data)
        else:
            self._dshape = dshape_create(dshape)
            assert dshape_table_check(self._dshape)
        if create and self._dshape is None:
            raise ValueError('Cannot create a table without a dshape')
        if self._dshape is None or (not create and metadata.ATTR_TABLE in self._storagegroup):
            self._load_table()
        else:
            self._create_table(fillvalues or {})
        if data is not None:
            self.append(data, indices=indices)

    @property
    def name(self):
        return self._name

    def _chunks_for(self, name):
        chunks = self._chunks
        if chunks is None:
            return None
        if isinstance(chunks, (integer_types, tuple)):
            return chunks
        if name in chunks:
            return chunks[name]
        if '*' in chunks:
            return chunks['*']
        return None

    @property
    def storagegroup(self):
        "Return the storagegroup form this column"
        return self._storagegroup

    def _load_table(self): #, scheduler):
        node = self._storagegroup
        if metadata.ATTR_TABLE not in node.attrs:
            raise ValueError('Group "%s" is not a Table', self.name)
        version = node.attrs[metadata.ATTR_VERSION]
        if version != metadata.VALUE_VERSION:
            raise ValueError('Invalid version "%s" for Table', version)
        nrow = node.attrs[metadata.ATTR_NROWS]
        self._dshape = dshape_create(node.attrs[metadata.ATTR_DATASHAPE])
        assert dshape_table_check(self._dshape)
        self._ids = IdColumn()
        self._ids.load_dataset(dshape=None, nrow=nrow)
        for (name, dshape) in dshape_fields(self._dshape):
            column = self._create_column(name)
            column.load_dataset(dshape=dshape,
                                nrow=nrow,
                                shape=dshape_to_shape(dshape))

    def _create_table(self, fillvalues):
        node = self.storagegroup
        node.attrs[metadata.ATTR_TABLE] = self.name
        node.attrs[metadata.ATTR_VERSION] = metadata.VALUE_VERSION
        node.attrs[metadata.ATTR_DATASHAPE] = str(self._dshape)
        node.attrs[metadata.ATTR_NROWS] = 0
        # create internal id dataset
        self._ids = IdColumn(storagegroup=self.storagegroup)
        self._ids.create_dataset(dshape=None, fillvalue=-1)
        for (name, dshape) in dshape_fields(self._dshape):
            assert name not in self._columndict
            shape = dshape_to_shape(dshape)
            fillvalue = fillvalues.get(name, None)
            chunks = self._chunks_for(name)
            #TODO compute chunks according to the shape
            column = self._create_column(name)
            column.create_dataset(dshape=dshape,
                                  chunks=chunks,
                                  fillvalue=fillvalue,
                                  shape=shape)

    def _create_column(self, name):
        column = Column(name, self._ids, storagegroup=self.storagegroup)
        index = len(self._columns)
        self._columndict[name] = index
        self._columns.append(column)
        return column

    @property
    def dshape(self):
        return self._dshape

    @property
    def ncol(self):
        return len(self._columns)

    @property
    def nrow(self):
        return len(self.index)

    def __contains__(self, colname):
        return colname in self._columndict

    def _resize_rows(self, newsize, index=None):
        self._ids.resize(newsize, index)

    def resize(self, newsize, index=None):
        self._resize_rows(newsize, index)
        self._storagegroup.attrs[metadata.ATTR_NROWS] = newsize
        for column in self._columns:
            column.resize(newsize)

    def _allocate(self, count, index=None):
        index = self._ids._allocate(count, index)
        newsize = self._ids.size
        for column in self._columns:
            column.resize(newsize)
        return index

    def touch_rows(self, loc=None):
        "Signals that the values at loc have been changed"
        self._ids.touch(loc)

    def __delitem__(self, col):
        raise NotImplementedError("Cannot delete column(s) '%s' yet", col)
        #del self._ids[key]

    def drop(self, index, locs=None):
        # for column in self._columns:
        #     val = column[index]
        #     val = column.fillvalue
        #     column[index] = val
        if locs is None:
            locs = self.index[index]
        self._ids._delete_ids(locs, index)
        #del
        # for column in self._columns:
        #    column.update()

    def parse_data(self, data, indices=None):
        if data is None:
            return None
        if isinstance(data, Mapping):
            if are_instances(data.values(), np.ndarray) or are_instances(data.values(), list):
                return data # Table can parse this
        if isinstance(data, (np.ndarray, Mapping)):
            # delegate creation of structured data to pandas for now
            data = pd.DataFrame(data, columns=self.columns, index=indices)
        return data # hoping it works

    def append(self, data, indices=None):
        """
        Append Table-like data to the Table.
        The data has to be compatible. It can be from multiple sources [more details needed].
        """
        if data is None:
            return
        data = self.parse_data(data, indices)
        dshape = dshape_extract(data)
        if not dshape_compatible(dshape, self.dshape):
            raise ValueError("{shape} incompatible data shape in append".format(shape=str(dshape)))
        length = -1
        all_arrays = True
        for colname in self:
            fromcol = data[colname]
            if length is -1:
                length = len(fromcol)
            elif length != len(fromcol):
                raise ValueError('Cannot append ragged values')
            all_arrays |= isinstance(fromcol, np.ndarray)
        if length == 0:
            return
        if indices is not None and len(indices) != length:
            raise ValueError('Bad index length (%d/%d)', len(indices), length)
        indices = self._allocate(length, indices)
        if all_arrays:
            indices = indices_to_slice(indices)
            for colname in self:
                tocol = self._column(colname)
                fromcol = data[colname]
                tocol[indices] = fromcol[0:length]
        else:
            for colname in self:
                tocol = self._column(colname)
                fromcol = data[colname]
                for i in range(length):
                    tocol[indices[i]] = fromcol[i]

    def add(self, row, index=None):
        "Add one row to the Table"
        assert len(row) == self.ncol

        if isinstance(row, Mapping):
            for colname in self:
                if colname not in row:
                    raise ValueError('Missing value for column "%s" from row'%colname)
        if index is None:
            index = self._allocate(1)
        else:
            index = self._allocate(1, [index])
        #start = self.id_to_index(index[0], as_slice=False)
        start = index[0]

        if isinstance(row, Mapping):
            for colname in self:
                tocol = self._column(colname)
                tocol[start] = row[colname]
        else:
            for colname, value in zip(self, row):
                tocol = self._column(colname)
                tocol[start] = value

    def binary(self, op, other, **kwargs):
        res = super(Table, self).binary(op, other, **kwargs)
        #return Table(None, data=res, scheduler=self.scheduler, create=True)
        return Table(None, data=res, create=True)

    @staticmethod
    def from_array(array, name=None, columns=None, offsets=None, dshape=None, **kwds):
        """offsets is a list of indices or pairs. """
        if offsets is None:
            offsets = [(i, i+1) for i in range(array.shape[1])]
        if offsets is not None:
            if all_int(offsets):
                offsets = [(offsets[i], offsets[i+1]) for i in range(len(offsets)-1)]
            elif not all([isinstance(elt, tuple) for elt in offsets]):
                raise ValueError('Badly formed offset list %s', offsets)

        if columns is None:
            if dshape is None:
                columns = gen_columns(len(offsets))
            else:
                dshape = dshape_create(dshape)
                columns = [fieldname for (fieldname, _) in dshape_fields(dshape)]
        if dshape is None:
            dshape_type = dshape_from_dtype(array.dtype)
            dims = ["" if (off[0]+1 == off[1]) else "%d *"%(off[1] - off[0])
                    for off in offsets]
            dshapes = ["%s: %s %s"%(column, dim, dshape_type)
                       for column, dim in zip(columns, dims)]
            dshape = "{" + ", ".join(dshapes) + "}"
        data = OrderedDict()
        for nam, off in zip(columns, offsets):
            if off[0]+1 == off[1]:
                data[nam] = array[:, off[0]]
            else:
                data[nam] = array[:, off[0]:off[1]]
        return Table(name, data=data, dshape=dshape, **kwds)

    def eval(self, expr, inplace=False, name=None, result_object=None, user_dict=None):
        """Evaluate the ``expr`` on columns and return the result.

        Args:
            inplace: boolean, optional
                Apply the changes in place
            name: string
                used when a new table/view is created, otherwise ignored
            result_object: string
               Posible values for result_object: {'raw_numexpr', 'index', 'view', 'table'}
               When expr is conditional.
               Note: a result as 'view' is not guaranteed: it may be 'table' when the calculated
               index is not sliceable
               - 'table' or None when expr is an assignment
               Default values for result_object :
               - 'indices' when expr is conditional
               - NA i.e. always None when inplace=True, otherwise a new table is returned
        """
        if inplace and result_object:
            raise ValueError("'inplace' and 'result_object' options are not compatible")
        if user_dict is None:
            context = {key: self[key].values for key in self.columns}
        else:
            context = user_dict
        is_assign = False
        try:
            res = ne.evaluate(expr, local_dict=context)
            if result_object is None:
                result_object = 'index'
        except SyntaxError as err:
            # maybe an assignment ?
            try:
                l_col, r_expr = expr.split('=', 1)
                l_col = l_col.strip()
                if l_col not in self.columns:
                    raise err
                res = ne.evaluate(r_expr.strip(), local_dict=context)
                is_assign = True
            except:
                raise err
            if result_object is not None and result_object != 'table':
                raise ValueError("result_object={} is not valid when expr "
                                 "is an assignment".format(result_object))
        else:
            if result_object not in ('raw_numexpr', 'index', 'view', 'table'):
                raise ValueError("result_object={} is not valid".format(result_object))
        if is_assign:
            if inplace:
                self[l_col] = res
                return
            # then not inplace ...
            def cval(key):
                return res if key == l_col else self[key].values
            data = [(cname, cval(cname)) for cname in self.columns]
            return Table(name=name, data=OrderedDict(data), indices=self.index)
        # not an assign ...
        if res.dtype != 'bool':
            raise ValueError(
                'expr must be an assignment '
                'or a conditional expr.!')
        if inplace:
            raise ValueError('inplace eval of conditional expr '
                             'not implemented!')
        if result_object == 'raw_numexpr':
            return res
        indices = np.where(res)[0]
        ix_slice = indices_to_slice(indices)
        if result_object == 'index':
            return ix_slice
        if result_object == 'view':
            return self.iloc[ix_slice, :]
        # as a new table ...
        data = [(cname, self[cname].values[indices]) for cname in self.columns]
        return Table(name=name,
                     data=OrderedDict(data),
                     indices=self._ids.values[indices])
