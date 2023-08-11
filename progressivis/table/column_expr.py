from __future__ import annotations

import logging

import numpy as np
import numexpr as ne
from progressivis.core.utils import integer_types
from .column_base import BasePColumn
from .dshape import dshape_create, DataShape
from .table_base import IndexPTable, BasePTable

from typing import Any, Tuple, List, Sequence

from ..core.types import Index

Shape = Tuple[int, ...]


logger = logging.getLogger(__name__)

__all__ = ["PColumnExpr"]


class FakeCol:
    def __init__(self, index: IndexPTable, shape: Shape) -> None:
        self.index = index
        self.dshape = shape
        self.dtype = shape

    @property
    def shape(self) -> Tuple[int]:
        return (len(self.index),)


class PColumnExpr(BasePColumn):
    def __init__(
        self,
        name: str,
        table: BasePTable,
        index: IndexPTable,
        expr: str,
        cols: List[str],
        dtype: np.dtype[Any],
        xshape: Shape = (),
        dshape: DataShape | None = None,
    ) -> None:
        """Create a new expression column."""
        super().__init__(name, index, base=None)
        self.table = table
        self.expr = expr
        self.cols = cols
        self._dtype: np.dtype[Any] = np.dtype(dtype)
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
        return self._dshape or dshape_create(str(self._dtype))

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
        res = ne.evaluate(self.expr, local_dict=context)
        return res

    def __getitem__(self, index: Index) -> Any:
        index = [index] if isinstance(index, integer_types) else index
        context = {
            k: self.table.to_array(locs=index, columns=[k]).reshape(-1)
            for k in self.cols
        }
        res = ne.evaluate(self.expr, local_dict=context)
        return res
