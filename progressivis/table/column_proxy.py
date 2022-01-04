"Proxy for column"
from __future__ import annotations

from .column_base import BaseColumn

from typing import Sequence, TYPE_CHECKING, Tuple, Any, List, cast

if TYPE_CHECKING:
    from .table_base import IndexTable
    from .dshape import DataShape
    import numpy as np


class ColumnProxy(BaseColumn):
    "Proxy class for a column"

    def __init__(self, base: BaseColumn, index: IndexTable) -> None:
        super(ColumnProxy, self).__init__(name=base.name, base=base, index=index)

    @property
    def chunks(self) -> Tuple[int, ...]:
        assert self._base is not None
        return self._base.chunks

    @property
    def shape(self) -> Tuple[int, ...]:
        assert self._base is not None
        return self._base.shape

    def set_shape(self, shape: Sequence[int]) -> None:
        assert self._base is not None
        self._base.set_shape(shape)

    def __delitem__(self, index: Any) -> None:
        raise RuntimeError("Cannot delete in %s" % type(self))

    @property
    def maxshape(self) -> Tuple[int, ...]:
        assert self._base is not None
        return self._base.maxshape

    @property
    def dtype(self) -> np.dtype[Any]:
        assert self._base is not None
        return self._base.dtype

    @property
    def dshape(self) -> DataShape:
        assert self._base is not None
        return self._base.dshape

    @property
    def size(self) -> int:
        assert self._base is not None
        return self._base.size

    def __len__(self) -> int:
        assert self._base is not None
        return len(self._base)

    @property
    def fillvalue(self) -> Any:
        assert self._base is not None
        return self._base.fillvalue

    @property
    def value(self) -> np.ndarray[Any, Any]:
        assert self._base is not None
        return cast(np.ndarray[Any, Any], self._base[:])

    def __getitem__(self, index: Any) -> Any:
        assert self._base is not None
        return self._base[index]

    def __setitem__(self, index: Any, val: Any) -> None:
        assert self._base is not None
        self._base[index] = val

    def resize(self, newsize: int) -> None:
        assert self._base is not None
        self._base.resize(newsize)

    def tolist(self) -> List[Any]:
        return list(self.values)
