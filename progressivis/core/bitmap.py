"""
Manage bit sets as ordered list of integers very efficiently,
relying on pyroaring RoaringBitmaps.
"""
from __future__ import annotations

import array

from pyroaring import BitMap, FrozenBitMap
import numpy as np
from collections.abc import Iterable

from typing import Optional, Any, Union, Iterator, overload

# pragma no cover
# pylint: disable=invalid-name
_integer_types = (int, np.integer)


class bitmap:
    # pylint: disable=invalid-name
    """
    Derive from an efficient and light-weight ordered set of 32 bits integers.
    """

    def __init__(self,
                 values: Any = None,
                 copy_on_write: bool = False,
                 optimize: bool = True):
        self.bm: BitMap
        if isinstance(values, bitmap):
            values = values.bm
        elif isinstance(values, slice):
            values = range(
                values.start, values.stop, (values.step if values.step else 1)
            )
        self.bm = BitMap(values, copy_on_write, optimize)

    # def __new__(cls, values=None, copy_on_write=False, optimize=True, no_init=False):
    #     if isinstance(values, slice):
    #         values = range(
    #             values.start, values.stop, (values.step if values.step else 1)
    #         )
    #     return super(bitmap, cls).__new__(cls, values, copy_on_write, optimize, no_init)

    def copy(self) -> bitmap:
        return bitmap(self.bm.copy())

    def clear(self) -> None:
        "Clear the bitmap in-place"
        self.bm.clear()

    def freeze(self) -> bitmap:
        return bitmap(FrozenBitMap(self.bm))

    def __contains__(self, other: Any) -> bool:
        if isinstance(other, _integer_types):
            return self.bm.__contains__(int(other))
        other = self.asbitmap(other)
        assert isinstance(other, bitmap)
        return other.bm <= self.bm

    def __repr__(self) -> str:
        length = len(self.bm)
        if length > 10:
            values = ", ".join([str(n) for n in self[0:6]])
            values += "...(%d)...%d)" % (length, self[length - 1])
        else:
            values = ", ".join([str(n) for n in self])
        return "bitmap([%s])" % values

    def __iter__(self) -> Iterator[int]:
        return iter(self.bm)

    def __eq__(self, value: Any, /) -> bool:
        if isinstance(value, bitmap):
            return self.bm == value.bm
        return False

    def __ne__(self, value: object, /) -> bool:
        if isinstance(value, bitmap):
            return self.bm != value.bm
        return True

    def __ge__(self, value: bitmap, /) -> bool:
        return self.bm >= value.bm

    def __gt__(self, value: bitmap, /) -> bool:
        return self.bm > value.bm

    def __le__(self, value: bitmap, /) -> bool:
        return self.bm <= value.bm

    def __lt__(self, value: bitmap, /) -> bool:
        return self.bm < value.bm

    def __hash__(self) -> int:
        return hash(self.bm)

    @overload
    def __getitem__(self, values: int) -> int:
        ...

    @overload
    def __getitem__(self, values: Union[Iterable[int], slice, bitmap]) -> bitmap:
        ...

    def __getitem__(self, values: Union[int, Iterable[int], slice, bitmap]) -> Union[int, bitmap]:
        bm: Union[bitmap, BitMap, Exception]
        if isinstance(values, int):
            return self.bm[values]
        elif isinstance(values, (Iterable, bitmap, BitMap)):
            bm = bitmap()
            for index in values:
                bm.add(self.bm[int(index)])
            return bm
        elif isinstance(values, slice):
            bm = self.bm[values]
        else:
            raise ValueError("Invalid index for bitmap")
        # Fixes a bug in BitMap
        if isinstance(bm, BitMap):
            return bitmap(bm)
        if isinstance(bm, Exception):
            raise bm
        raise RuntimeError("Should not happen")

    def __len__(self) -> int:
        return len(self.bm)

    def min(self) -> int:
        return self.bm.min()

    def max(self) -> int:
        return self.bm.max()

    def add(self, value: int) -> None:
        self.bm.add(value)

    def any(self) -> bool:
        return bool(self.bm)

    def to_array(self) -> array.array[int]:
        return self.bm.to_array()

    def update(self, values: Any) -> None:
        """
        Add new values from either a bitmap, an array, a slice, or an Iterable
        """
        try:
            self.bm.update(self, values)  # type: ignore
        except TypeError:
            if values is None:
                return
            # NP check the copy here for slice
            if isinstance(values, slice):
                values = range(*values.indices(values.stop + 1))
            # do not call bitmap constructor here cause
            # BitMap constructor calls update=>infinite recursion
            values = array.array("I", values)
            self.bm.update(self, values)  # type: ignore

    def pop(self, length: int = 1) -> bitmap:
        "Remove one or many items and return them as a bitmap"
        if length >= len(self):
            ret = bitmap(self)
            self.clear()
            return ret
        ret = self[0:length]
        self -= ret
        return ret

    def remove(self, value: int) -> None:
        self.bm.remove(value)

    def remove_range(self, start: int, stop: int) -> None:
        self.bm.remove_range(start, stop)

    def symmetric_difference(self, other: bitmap) -> bitmap:
        return bitmap(self.bm.symmetric_difference(other.bm))

    def symmetric_difference_update(self, other: bitmap) -> None:
        self.bm.symmetric_difference_update(other.bm)

    def contains_range(self, start: int, end: int) -> bool:
        return self.bm.contains_range(start, end)

    @staticmethod
    def difference(*bitmaps: bitmap) -> bitmap:
        return bitmap(BitMap.difference(*[b.bm for b in bitmaps]))

    def difference_cardinality(self, other: bitmap) -> int:
        return self.bm.difference_cardinality(other.bm)

    def to_slice_maybe(self) -> Union[slice, bitmap]:
        "Convert this bitmap to a slice if possible, or return self"
        length = len(self)
        if length == 0:
            return slice(0, 0)
        first = self.min()
        last = self.max()
        if last - first + 1 == length:
            return slice(first, last + 1)
        return self

    @staticmethod
    def asbitmap(x: Any) -> bitmap:
        "Try to coerce the value as a bitmap"
        if x is None:
            return bitmap()
        if isinstance(x, bitmap):
            return x
        if isinstance(x, _integer_types):
            return bitmap([x])
        return bitmap(x)

    def __or__(self, other: Optional[bitmap]) -> bitmap:
        if other is None:
            return bitmap(self.bm)
        if isinstance(other, bitmap):
            return bitmap(BitMap.__or__(self.bm, other.bm))
        raise TypeError("Invalid type {type(other)}")

    def __and__(self, other: Optional[bitmap]) -> bitmap:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, bitmap):
            return bitmap(BitMap.__and__(self.bm, other.bm))
        raise TypeError("Invalid type {type(other)}")

    def __xor__(self, other: Optional[bitmap]) -> bitmap:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, bitmap):
            return bitmap(BitMap.__xor__(self.bm, other.bm))
        raise TypeError("Invalid type {type(other)}")

    def __sub__(self, other: Optional[bitmap]) -> bitmap:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, bitmap):
            return bitmap(BitMap.__sub__(self.bm, other.bm))
        raise TypeError("Invalid type {type(other)}")

    def __ior__(self, other: Optional[bitmap]) -> bitmap:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, bitmap):
            BitMap.__ior__(self.bm, other.bm)
            return self
        raise TypeError("Invalid type {type(other)}")

    def __iand__(self, other: Optional[bitmap]) -> bitmap:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, bitmap):
            BitMap.__iand__(self.bm, other.bm)
            return self
        raise TypeError("Invalid type {type(other)}")

    def __ixor__(self, other: Optional[bitmap]) -> bitmap:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, bitmap):
            BitMap.__ixor__(self.bm, other.bm)
            return self
        raise TypeError("Invalid type {type(other)}")

    def __isub__(self, other: Optional[bitmap]) -> bitmap:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, bitmap):
            BitMap.__isub__(self.bm, other.bm)
            return self
        raise TypeError("Invalid type {type(other)}")

    def flip(self, start: int, end: int) -> bitmap:
        """
        Compute the negation of the bitmap within the specified interval.
        """
        return bitmap(BitMap.flip(self.bm, start, end))

    @staticmethod
    def union(*bitmaps: bitmap) -> bitmap:
        """
        Return the union of the bitmaps.
        """
        bm = BitMap.union(*[b.bm for b in bitmaps])
        return bitmap(bm) if isinstance(bm, BitMap) else bm

    @staticmethod
    def intersection(*bitmaps: bitmap) -> bitmap:
        """
        Return the intersection of the bitmaps.
        """
        bm = BitMap.intersection(*[b.bm for b in bitmaps])
        return bitmap(bm) if isinstance(bm, BitMap) else bm

    @staticmethod
    def deserialize(buff: bytes) -> bitmap:
        """
        Generate a bitmap from the given serialization.
        """
        return bitmap(BitMap.deserialize(buff))

    def serialize(self) -> bytes:
        return self.bm.serialize()


NIL_BITMAP = bitmap(FrozenBitMap())
