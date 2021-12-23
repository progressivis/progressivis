from __future__ import annotations

from typing import Union, Optional, Any, TYPE_CHECKING, cast, List

from collections import Iterable
import logging

import numpy as np

from progressivis.core.utils import integer_types
from progressivis.core.bitmap import bitmap
from .base import Dataset
from .hierarchy import AttributeImpl


if TYPE_CHECKING:
    from .base import Shape, DTypeLike, ArrayLike, Attribute

logger = logging.getLogger(__name__)


class RangeError(RuntimeError):
    pass


class RangeDataset(Dataset):
    def __init__(
        self,
        name: str,
        shape: Optional[Shape] = None,
        dtype: Optional[DTypeLike] = None,
        data: Optional[Any] = None,
        **kwds,
    ):
        self._name = name
        if shape is None:
            if data is not None and hasattr(data, "shape"):
                shape = data.shape
            else:
                shape = (0,)
        if len(shape) != 1:
            raise ValueError(
                "RangeDataset shape should be one-dimensional and not %s", shape
            )
        self._shape = shape
        if dtype is not None:
            if dtype != np.int_:
                raise TypeError("dtype of a RangeDataset should be integer")
        self._dtype: np.dtype = np.dtype(np.int_)
        if kwds:
            logger.warning("Unused arguments %s", kwds)
        self._attrs = AttributeImpl()

    def flush(self) -> None:
        pass

    def close_all(self, recurse=True) -> None:
        pass

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def maxshape(self) -> Shape:
        return self._shape

    @property
    def fillvalue(self) -> Any:
        return 0

    @property
    def chunks(self) -> Shape:
        return self._shape

    @property
    def size(self) -> int:
        return self._shape[0]

    def resize(self, size: Union[int, ArrayLike], axis: Optional[int] = None) -> None:
        if isinstance(size, tuple):
            if len(size) != 1:
                raise KeyError(
                    "Invalid size tuple, should have dim=1 instead of %s", size
                )
            size = size[0]
        if size == self._shape[0]:
            return
        self._shape = (size,)

    def __getitem__(self, args) -> Any:
        if isinstance(args, tuple):
            if len(args) != 1:
                raise KeyError("too many dimensions in __getitem__: %s", args)
            args = args[0]
        if isinstance(args, integer_types):
            if args < self.size:
                return args
            else:
                raise IndexError("Index %d out of bounds for size %d", args, self.size)
        elif isinstance(args, np.ndarray):
            if args.dtype == np.int_:
                if (args >= self.size).any():
                    raise IndexError(
                        "Some index in %s out of bounds for size %d", args, self.size
                    )
                return args
            elif args.dtype == np.bool_:
                return self[np.where(args)[0]]
        elif isinstance(args, Iterable):
            try:
                count = len(cast(List, args))
            # pylint: disable=bare-except
            except Exception:
                count = -1
            return self[np.fromiter(args, dtype=np.int64, count=count)]
        elif isinstance(args, slice):
            return np.arange(*args.indices(self.size), dtype=np.int64)
        elif isinstance(args, bitmap):
            if args.max() >= self.size:
                raise IndexError(
                    "Some index in %s out of bounds for size %d", args, self.size
                )
            return args
        raise KeyError("Invalid key for __getitem__: %s", args)

    def __setitem__(self, args, val) -> None:
        if not np.array_equal(self[args], np.asarray(val)):
            raise RangeError("values incompatible with range")

    def __len__(self) -> int:
        return self._shape[0]

    @property
    def attrs(self) -> Attribute:
        return self._attrs

    @property
    def name(self) -> str:
        return self._name
