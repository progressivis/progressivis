from __future__ import annotations

import logging

import numpy as np

from .column_proxy import ColumnProxy
from ..core.utils import integer_types
from .dshape import dataframe_dshape, DataShape
from typing import TYPE_CHECKING, Tuple, Any, Sequence, Callable, Optional

if TYPE_CHECKING:
    from .column_base import BaseColumn
    from .table_base import IndexTable

Shape = Tuple[int, ...]

logger = logging.getLogger(__name__)


class ColumnSelectedView(ColumnProxy):
    def __init__(self, base: BaseColumn, index: IndexTable):
        super(ColumnSelectedView, self).__init__(base, index=index)

    @property
    def shape(self) -> Tuple[int, ...]:
        assert self._base is not None
        tshape = list(self._base.shape)
        tshape[0] = len(self)
        return tuple(tshape)

    def set_shape(self, shape: Sequence[int]) -> None:
        raise RuntimeError("set_shape not implemented for %s", type(self))

    @property
    def maxshape(self) -> Tuple[int, ...]:
        assert self._base is not None
        tshape = list(self._base.maxshape)
        tshape[0] = len(self)
        return tuple(tshape)

    def __len__(self) -> int:
        assert self.index is not None
        return self.index.last_id + 1

    @property
    def value(self) -> np.ndarray[Any, Any]:
        assert self._base is not None
        return self._base[self.index.id_to_index(slice(None))]  # type: ignore

    def __getitem__(self, index: Any) -> Any:
        assert self._base is not None
        if isinstance(index, integer_types):
            return self._base[index]
        bm = self.index._any_to_bitmap(index)
        return self._base[bm]


class ColumnComputedView(ColumnSelectedView):
    def __init__(
        self,
        base: BaseColumn,
        index: IndexTable,
        aka: str,
        func: Callable,
        dtype: Optional[np.dtype] = None,
        xshape: Shape = (),
    ):
        super().__init__(base, index=index)
        self.aka = aka
        self.func = func
        self._is_ufunc = isinstance(func, (np.ufunc, np.vectorize))
        self._otype = dtype
        self._xshape = xshape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._otype if self._otype is not None else super().dtype  # type: ignore

    @property
    def shape(self) -> Shape:
        return (len(self._index), *self._xshape)

    @property
    def dshape(self) -> DataShape:
        return (
            dataframe_dshape(self._otype) if self._otype is not None else super().dshape
        )  # type: ignore

    @property
    def name(self):
        return self.aka

    @property
    def value(self) -> np.ndarray[Any, Any]:
        res = super().value
        return self.func(res)  # type: ignore

    def __getitem__(self, index: Any) -> Any:
        raw_index = index
        index = [index] if isinstance(index, integer_types) else index
        values = super().__getitem__(index)
        if self._is_ufunc:
            ret = self.func(values)  # type: ignore
        else:
            ret = np.apply_along_axis(
                self.func, len(values.shape) - 1, np.array(values)
            )
        if isinstance(raw_index, integer_types):
            return ret[0]
        return ret
