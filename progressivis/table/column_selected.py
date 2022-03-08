from __future__ import annotations

import logging

import numpy as np

from .column_proxy import ColumnProxy

from typing import TYPE_CHECKING, Tuple, Any, Sequence

if TYPE_CHECKING:
    from .column_base import BaseColumn
    from .table_base import IndexTable


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
        bm = self.index._any_to_bitmap(index)
        assert self._base is not None
        return self._base[bm]
