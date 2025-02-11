from __future__ import annotations

import logging

import numpy as np

from .column_proxy import PColumnProxy
from ..core.utils import integer_types
from .dshape import dataframe_dshape, dshape_create, DataShape
from typing import TYPE_CHECKING, Tuple, Any, Sequence, Callable

if TYPE_CHECKING:
    from .column_base import BasePColumn
    from .table_base import IndexPTable

Shape = Tuple[int, ...]

logger = logging.getLogger(__name__)


class PColumnSelectedView(PColumnProxy):
    def __init__(self, base: BasePColumn, index: IndexPTable):
        super().__init__(base, index=index)

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
        bm = self.index._any_to_pintset(index)
        return self._base[bm]


class PColumnComputedView(PColumnSelectedView):
    def __init__(
        self,
        base: BasePColumn,
        index: IndexPTable,
        aka: str,
        func: Callable[..., Any],
        dtype: np.dtype[Any] | None = None,
        xshape: Shape = (),
    ) -> None:
        super().__init__(base, index=index)
        self.aka = aka
        self.func = func
        self._is_ufunc = isinstance(func, (np.ufunc, np.vectorize))
        self._otype = dtype
        self._xshape = xshape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._otype if self._otype is not None else super().dtype

    @property
    def shape(self) -> Shape:
        return (len(self._index), *self._xshape)

    @property
    def dshape(self) -> DataShape:
        return (
            dshape_create(dataframe_dshape(self._otype))
            if self._otype is not None else super().dshape
        )

    @property
    def name(self) -> str:
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
            ret = self.func(values)
        else:
            ret = np.apply_along_axis(
                self.func, len(values.shape) - 1, np.array(values)
            )
        if isinstance(raw_index, integer_types):
            if ret.shape:
                return ret[0]
            else:
                return ret[()]
        return ret
