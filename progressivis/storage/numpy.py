from __future__ import annotations


import logging

import numpy as np

from progressivis.core.utils import integer_types

try:
    from progressivis.utils.fast import next_pow2
except ImportError:
    from progressivis.core.utils import next_pow2
from .base import StorageEngine, Dataset, Group
from .hierarchy import GroupImpl, AttributeImpl


from typing import Union, Optional, Any, Tuple, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .base import Attribute
    from numpy.typing import DTypeLike, ArrayLike

Shape = Tuple[int, ...]


logger = logging.getLogger(__name__)


class NumpyDataset(Dataset):
    def __init__(
        self,
        name: str,
        shape: Optional[Shape] = None,
        dtype: Optional[DTypeLike] = None,
        data: Optional[Any] = None,
        **kwds: Any
    ) -> None:
        self._name = name
        self.base: np.ndarray[Any, Any]
        if data is not None:
            self.base = np.array(data, dtype=dtype)
        else:
            assert shape is not None
            self.base = np.empty(shape=shape, dtype=dtype)
        if "maxshape" in kwds:
            del kwds["maxshape"]
        if "fillvalue" in kwds:
            self._fillvalue = kwds.pop("fillvalue")
        else:
            if isinstance(self.base.dtype, np.integer):
                self._fillvalue = 0
            else:
                self._fillvalue = np.nan
        if kwds:
            logger.warning("Ignored keywords in NumpyDataset: %s", kwds)
        self.view: np.ndarray[Any, np.dtype[Any]] = self.base
        self._attrs = AttributeImpl()

    @property
    def shape(self) -> Shape:
        return cast(Shape, self.view.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.view.dtype

    @property
    def maxshape(self) -> Shape:
        return cast(Shape, self.view.shape)

    @property
    def fillvalue(self) -> Any:
        return self._fillvalue

    @property
    def chunks(self) -> Shape:
        return cast(Shape, self.view.shape)

    @property
    def size(self) -> int:
        return cast(int, self.view.shape[0])

    def resize(self, size: Union[int, ArrayLike], axis: Optional[int] = None) -> None:
        if isinstance(size, integer_types):
            size = np.array(tuple([size] + list(self.base.shape[1:])))
        else:
            size = np.array(size)
        baseshape = np.array(self.base.shape)
        viewshape = self.view.shape
        if (size > baseshape).any():
            # self.view = None
            newsize = []
            for s, shape in zip(size, baseshape):
                if s > shape:
                    s = next_pow2(s)
                newsize.append(s)
            self.base = np.resize(self.base, tuple(newsize))
        # fill new areas with fillvalue
        if (size > viewshape).any() and (size != 0).all():
            newarea = [np.s_[0:os] for os in viewshape]
            for i in range(len(viewshape)):
                s = size[i]
                os = viewshape[i]
                if s > os:
                    newarea[i] = np.s_[os:s]
                    self.base[tuple(newarea)] = self._fillvalue
                newarea[i] = np.s_[0:s]
        else:
            newarea = [np.s_[0:s] for s in size]
        self.view = self.base[tuple(newarea)]

    def __getitem__(self, args: Any) -> Any:
        return self.view[args]

    def __setitem__(self, args: Any, val: Any) -> None:
        self.view[args] = val

    def __len__(self) -> int:
        return cast(int, self.view.shape[0])

    @property
    def attrs(self) -> Attribute:
        return self._attrs

    @property
    def name(self) -> str:
        return self._name

    def flush(self) -> None:
        pass


class NumpyGroup(GroupImpl):
    def __init__(self, name: str = "numpy", parent: Optional[GroupImpl] = None) -> None:
        super(NumpyGroup, self).__init__(name, parent=parent)

    def create_dataset(
        self,
        name: str,
        shape: Optional[Shape] = None,
        dtype: Optional[DTypeLike] = None,
        data: Optional[Any] = None,
        **kwds: Any
    ) -> Dataset:
        if name in self.dict:
            raise KeyError("name %s already defined", name)
        chunks = kwds.pop("chunks", None)
        if chunks is None:
            chunklen = None
        elif isinstance(chunks, integer_types):
            chunklen = int(chunks)
        elif isinstance(chunks, tuple):
            chunklen = 1
            for m in chunks:
                chunklen *= m
        if dtype is not None:
            dtype = np.dtype(dtype)
        fillvalue = kwds.pop("fillvalue", None)
        if fillvalue is None:
            if dtype == np.dtype(object):
                fillvalue = ""
            else:
                fillvalue = 0
        if data is None:
            if shape is None:
                data = np.ndarray([], dtype=dtype)
                shape = data.shape
            elif fillvalue == 0:
                data = np.zeros(shape, dtype=dtype)
            else:
                data = np.full(shape, fillvalue, dtype=dtype)

        arr = NumpyDataset(
            name, data=data, shape=shape, dtype=dtype, fillvalue=fillvalue, **kwds
        )
        self.dict[name] = arr
        return arr

    def _create_group(self, name: str, parent: Optional[GroupImpl]) -> Group:
        return NumpyGroup(name, parent=parent)


class NumpyStorageEngine(StorageEngine, NumpyGroup):
    def __init__(self) -> None:
        StorageEngine.__init__(self, "numpy")
        NumpyGroup.__init__(self, "/", None)

    def open(self, name: str, flags: Any, **kwds: Any) -> None:
        pass

    def close(self, name: str, flags: Any, **kwds: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def __contains__(self, name: str) -> bool:
        return NumpyGroup.__contains__(self, name)

    @staticmethod
    def create_group(name: str  = "numpy", create: bool = True) -> Group:
        _ = create  # for pylint
        return NumpyGroup(name)
