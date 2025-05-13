from __future__ import annotations

from collections.abc import Iterable
import logging

import numpy as np
from progressivis.storage import Group, Dataset
from progressivis.core.utils import integer_types, get_random_name

# try:
#     from progressivis.utils.fast import indices_to_slice
# except ImportError:

from progressivis.core.utils import indices_to_slice

from .column_base import BasePColumn
from .dshape import dshape_to_h5py, np_dshape, dshape_create, DataShape, EMPTY_DSHAPE
from . import metadata
from .table_base import IndexPTable
from ..core.pintset import PIntSet

from typing import Any, Optional, Union, Tuple

from ..core.types import Chunks, Index, Shape


logger = logging.getLogger(__name__)

__all__ = ["PColumn"]


class PColumn(BasePColumn):
    def __init__(
        self,
        name: str,
        index: Optional[IndexPTable],
        base: Optional[BasePColumn] = None,
        storagegroup: Optional[Group] = None,
        dshape: Optional[Union[None, DataShape, str]] = None,
        fillvalue: Optional[Any] = None,
        shape: Optional[Shape] = None,
        chunks: Optional[Chunks] = None,
        indices: Optional[Index] = None,
        data: Optional[Any] = None,
    ) -> None:
        """Create a new column.

        if index is None and self.index return None, a new index and
        dataset are created.
        """
        indexwasnone: bool = index is None
        if index is None:
            if data is not None:  # check before creating everything
                length = len(data)
                if indices and length != len(indices):
                    raise ValueError("Bad index length (%d/%d)", len(indices), length)
            index = IndexPTable()
        super().__init__(name, index, base=base)
        if storagegroup is None:
            if index is not None and hasattr(index, "storagegroup"):
                # i.e. isinstance(index, PTable)
                storagegroup = getattr(index, "storagegroup")
                assert isinstance(storagegroup, Group)
            else:
                assert Group.default
                storagegroup = Group.default(name=get_random_name("column_"))
        self._storagegroup = storagegroup
        self.dataset: Optional[Dataset] = None
        self._dshape: DataShape = EMPTY_DSHAPE
        if isinstance(dshape, DataShape):
            self._dshape = dshape
        elif isinstance(dshape, str):
            self._dshape = dshape_create(dshape)
        if indexwasnone:
            self._complete_column(dshape, fillvalue, shape, chunks, data)
            if data is not None:
                self.append(data, indices)

    @property
    def storagegroup(self) -> Group:
        return self._storagegroup

    def _allocate(self, count: int, indices: Any = None) -> PIntSet:
        start = self.index.last_id + 1
        if indices is not None:
            ret = self.index._any_to_pintset(indices)
            assert ret
            if ret & self.index.index:
                raise ValueError("Indices contain duplicates")
            newsize = start + ret.max() + 1
        else:
            newsize = start + count
            ret = PIntSet(range(start, newsize))
        self._resize(newsize)
        self.index._resize_rows(newsize, ret)
        return ret

    def append(self, data: Any, indices: Optional[Any] = None) -> None:
        if data is None:
            return
        length = len(data)
        is_array = isinstance(data, (np.ndarray, list, BasePColumn))
        if indices is not None and len(indices) != length:
            raise ValueError("Bad index length (%d/%d)", len(indices), length)
        indices = self._allocate(len(data), indices)
        if is_array:
            indices = indices_to_slice(indices)
            self[indices] = data[0:length]
        else:
            for i in range(length):
                self[indices[i]] = data[i]

    def add(self, value: Any, index: Optional[Any] = None) -> None:
        if index is None:
            index = self._allocate(1)
        else:
            index = self._allocate(1, [index])
        index = index[0]
        self[index] = value

    def _complete_column(
        self,
        dshape: Optional[Union[str, DataShape]],
        fillvalue: Any,
        shape: Optional[Shape],
        chunks: Optional[Chunks],
        data: Any,
    ) -> None:
        if dshape is None:
            if data is None:
                raise ValueError(
                    'Cannot create column "%s" without dshape nor data', self.name
                )
            elif hasattr(data, "dshape"):
                dshape = data.dshape
            elif hasattr(data, "dtype"):
                dshape = np_dshape(data)
            else:
                raise ValueError(
                    'Cannot create column "%s" from data %s', self.name, data
                )
        dshape = dshape_create(dshape)  # make sure it is valid
        if shape is None and data is not None:
            shape = dshape.shape
        self.create_dataset(
            dshape=dshape, fillvalue=fillvalue, shape=shape, chunks=chunks
        )

    def create_dataset(
        self,
        dshape: Union[str, DataShape],
        fillvalue: Any,
        shape: Optional[Shape] = None,
        chunks: Optional[Chunks] = None,
    ) -> Dataset:
        dshape = dshape_create(dshape)  # make sure it is valid
        self._dshape = dshape
        dtype = dshape_to_h5py(dshape)
        maxshape: Tuple[Any, ...]
        if shape is None:
            maxshape = (None,)
            shape = dshape.shape
            shape = (0,)
            if chunks is None:
                chunks = (128 * 1024 // np.dtype(dtype).itemsize,)
        else:
            maxshape = tuple((None,) + tuple(shape))
            shape = tuple([0] + [0 if s is None else s for s in shape])
            if chunks is None:
                dims = list(shape)[1:]
                # count 16 entries for each variable dimension
                # TODO find a smarter way to allocate chunk size
                chk = [64]
                for d in dims:
                    chk.append(d if d != 0 else 64)
                chunks = tuple(chk)
        if isinstance(chunks, int):
            chunks = tuple([chunks])
        logger.debug(
            "column=%s, shape=%s, chunks=%s, dtype=%s",
            self._name,
            shape,
            chunks,
            str(dtype),
        )

        group = self._storagegroup
        if self.name in group:
            logger.warning('Deleting dataset named "%s"', self.name)
            del group[self.name]
        dataset = group.create_dataset(
            self.name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            maxshape=maxshape,
            fillvalue=fillvalue,
        )
        dataset.attrs[metadata.ATTR_COLUMN] = True
        dataset.attrs[metadata.ATTR_VERSION] = metadata.VALUE_VERSION
        dataset.attrs[metadata.ATTR_DATASHAPE] = str(dshape)
        self.dataset = dataset
        dataset.flush()
        return dataset

    def load_dataset(
        self,
        dshape: DataShape,
        nrow: int,
        shape: Optional[Shape] = None,
        is_id: bool = False,
    ) -> Optional[Dataset]:
        self._dshape = dshape
        if shape is None:
            shape = (nrow,)
        else:
            shape = tuple((nrow,) + tuple(shape))
        dtype = dshape_to_h5py(dshape)
        group = self._storagegroup
        if is_id and self.name not in group:  # for lazy ID column creation
            return None
        dataset = group.require_dataset(self.name, dtype=dtype, shape=shape)
        assert (
            dataset.attrs[metadata.ATTR_COLUMN] is True
            and dataset.attrs[metadata.ATTR_VERSION] == metadata.VALUE_VERSION
            and dataset.attrs[metadata.ATTR_DATASHAPE] == str(dshape)
        )
        self.dataset = dataset
        return dataset

    @property
    def chunks(self) -> Tuple[int, ...]:
        assert self.dataset is not None
        return self.dataset.chunks

    @property
    def shape(self) -> Tuple[int, ...]:
        assert self.dataset is not None
        return self.dataset.shape

    def set_shape(self, shape: Shape) -> None:
        assert self.dataset is not None
        if not isinstance(shape, list):
            shape = list(shape)
        myshape = list(self.shape[1:])
        if len(myshape) != len(shape):
            raise ValueError(
                "Specified shape (%s) does not match colum shape (%s)"
                % (shape, myshape)
            )
        if myshape == shape:
            return
        logger.debug("Changing size from (%s) to (%s)", myshape, shape)
        self.dataset.resize(tuple([len(self)] + shape))

    @property
    def maxshape(self) -> Tuple[int, ...]:
        assert self.dataset is not None
        return self.dataset.maxshape

    @property
    def dtype(self) -> np.dtype[Any]:
        assert self.dataset is not None
        return self.dataset.dtype

    @property
    def dshape(self) -> DataShape:
        assert self.dataset is not None
        return self._dshape

    @property
    def size(self) -> int:
        assert self.dataset is not None
        return self.dataset.size

    def __len__(self) -> int:
        assert self.dataset is not None
        return len(self.index)

    @property
    def fillvalue(self) -> Any:
        assert self.dataset is not None
        return self.dataset.fillvalue

    @property
    def value(self) -> Any:
        assert self.dataset is not None
        return self.dataset[self.index.index.to_slice_maybe()]

    def __getitem__(self, index: Index) -> Any:
        assert self.dataset is not None
        if isinstance(index, np.ndarray):
            index = list(index)
        try:  # EAFP
            return self.dataset[index]
        except TypeError:
            if isinstance(index, Iterable):
                return np.array([self.dataset[e] for e in index])
            raise

    def read_direct(
        self,
        array: np.ndarray[Any, Any],
        source_sel: Optional[Any] = None,
        dest_sel: Optional[Any] = None,
    ) -> None:
        assert self.dataset is not None
        if hasattr(self.dataset, "read_direct"):
            if isinstance(source_sel, np.ndarray) and source_sel.dtype == np.int_:
                source_sel = list(source_sel)
            #            if is_fancy(source_sel):
            #                source_sel = fancy_to_mask(source_sel, self.shape)
            self.dataset.read_direct(array, source_sel, dest_sel)
        else:
            super().read_direct(array, source_sel, dest_sel)

    def __setitem__(self, index: Index, val: Any) -> None:
        assert self.dataset is not None
        if isinstance(index, integer_types):
            self.dataset[index] = val
        else:
            if hasattr(val, "values") and isinstance(val.values, np.ndarray):
                val = val.values
            if not hasattr(val, "shape"):
                val = np.asarray(val, dtype=self.dtype)

            if isinstance(index, np.ndarray) and index.dtype == np.int_:
                index = list(index)
            try:
                self.dataset[index] = val
            except TypeError:
                # TODO distinguish between unsupported fancy indexing and real error
                if isinstance(index, Iterable):
                    if isinstance(val, (np.ndarray, list)):
                        for e in index:
                            self.dataset[e] = val[e]
                    else:
                        for e in index:
                            self.dataset[e] = val
                else:
                    raise
        self.index.touch(index)

    def _resize(self, newsize: int) -> None:
        assert isinstance(newsize, integer_types)
        if self.size == newsize:
            return
        shape = self.shape
        assert self.dataset is not None
        if len(shape) == 1:
            self.dataset.resize((newsize,))
        else:
            shape = tuple([newsize] + list(shape[1:]))
            self.dataset.resize(shape)

    def resize(self, newsize: int) -> None:
        self._resize(newsize)
        if self.index is not None:
            self.index._resize_rows(newsize)

    def __delitem__(self, index: Index) -> None:
        assert self.dataset is not None
        del self.index[index]
        self.dataset[index] = self.fillvalue  # cannot propagate that to other columns
        self.dataset.resize(self.index.size)
