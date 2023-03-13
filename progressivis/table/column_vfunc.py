from __future__ import annotations

import logging

import numpy as np
from progressivis.core.utils import integer_types
from .column_base import BasePColumn
from .dshape import dshape_create, DataShape
from .table_base import IndexPTable, BasePTable

from typing import Any, Optional, Sequence, Union, Tuple, List, Callable

from ..core.types import Index

Shape = Tuple[int, ...]


logger = logging.getLogger(__name__)

__all__ = ["PColumnVFunc"]


class PColumnVFunc(BasePColumn):
    def __init__(
        self,
        name: str,
        table: BasePTable,
        index: IndexPTable,
        func: Callable[..., Any],
        cols: Union[str, List[str]],
        dtype: Union[np.dtype[Any], str],
        xshape: Shape = (),
        dshape: Optional[str] = None,
    ) -> None:
        """Create a column controlled by a self-sufficient, vectorized function."""
        super().__init__(name, index, base=None)
        self.table = table
        self.func = func
        self.cols = cols
        self._dtype: np.dtype[Any] = np.dtype(dtype)
        assert isinstance(xshape, tuple)
        self._xshape = xshape
        self._dshape = dshape

    @property
    def chunks(self) -> Tuple[int, ...]:
        return (1,)

    @property
    def shape(self) -> Shape:
        return (len(self._index), *self._xshape)

    def set_shape(self, shape: Sequence[int]) -> None:
        raise RuntimeError("Cannot set shape on %s" % type(self))

    def __delitem__(self, index: Any) -> None:
        raise RuntimeError("Cannot delete in %s" % type(self))

    @property
    def maxshape(self) -> Shape:
        return self.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def dshape(self) -> DataShape:
        return self._dshape or dshape_create(str(self._dtype))  # type: ignore

    @property
    def size(self) -> int:
        return len(self._index)

    def __len__(self) -> int:
        return len(self._index)

    @property
    def fillvalue(self) -> Any:
        return None

    def __setitem__(self, index: Any, val: Any) -> None:
        raise RuntimeError("Cannot set %s" % type(self))

    def resize(self, newsize: int) -> None:
        pass

    @property
    def value(self) -> Any:
        context = {
            k: self.table.to_array(locs=self._index.index, columns=[k]).reshape(-1)
            for k in self.cols
        }
        res = self.func(self._index.index, local_dict=context)
        return res

    def __getitem__(self, index: Index) -> Any:
        cols = self.cols if isinstance(self.cols, list) else [self.cols]
        raw_index = index
        index = [index] if isinstance(index, integer_types) else index
        context = {
            k: self.table[k].loc[index] for k in cols
        }
        res = self.func(raw_index, local_dict=context)
        return res
