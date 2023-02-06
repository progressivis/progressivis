"""
Manage bit sets as ordered list of integers very efficiently,
relying on pyroaring RoaringBitmaps.
"""
from __future__ import annotations

import array

from pyroaring import BitMap, FrozenBitMap
import numpy as np


from typing import Optional, Any, Union, Iterator, overload, Iterable

# pragma no cover
# pylint: disable=invalid-name
_integer_types = (int, np.integer)


class PIntSet(Iterable[int]):
    # pylint: disable=invalid-name
    """
    Derive from an efficient and light-weight ordered set of 32 bits integers.
    """

    def __init__(
        self, values: Any = None, copy_on_write: bool = False, optimize: bool = True
    ):
        self.bm: BitMap
        if isinstance(values, PIntSet):
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
    #     return super(PIntSet, cls).__new__(cls, values, copy_on_write, optimize, no_init)

    def copy(self) -> PIntSet:
        return PIntSet(self.bm.copy())

    def clear(self) -> None:
        "Clear the PIntSet in-place"
        self.bm.clear()

    def freeze(self) -> PIntSet:
        return PIntSet(FrozenBitMap(self.bm))

    def __contains__(self, other: Any) -> bool:
        if isinstance(other, _integer_types):
            return self.bm.__contains__(int(other))
        other = self.aspintset(other)
        assert isinstance(other, PIntSet)
        return other.bm <= self.bm

    def __repr__(self) -> str:
        length = len(self.bm)
        if length > 10:
            values = ", ".join([str(n) for n in self[0:6]])
            values += "...(%d)...%d)" % (length, self[length - 1])
        else:
            values = ", ".join([str(n) for n in self])
        return "PIntSet([%s])" % values

    def __iter__(self) -> Iterator[int]:
        return iter(self.bm)

    def __eq__(self, value: Any, /) -> bool:
        if isinstance(value, PIntSet):
            return self.bm == value.bm
        return False

    def __ne__(self, value: object, /) -> bool:
        if isinstance(value, PIntSet):
            return self.bm != value.bm
        return True

    def __ge__(self, value: PIntSet, /) -> bool:
        return self.bm >= value.bm

    def __gt__(self, value: PIntSet, /) -> bool:
        return self.bm > value.bm

    def __le__(self, value: PIntSet, /) -> bool:
        return self.bm <= value.bm

    def __lt__(self, value: PIntSet, /) -> bool:
        return self.bm < value.bm

    def __hash__(self) -> int:
        return hash(self.bm)

    @overload
    def __getitem__(self, values: int) -> int:
        ...

    @overload
    def __getitem__(self, values: Union[Iterable[int], slice, PIntSet]) -> PIntSet:
        ...

    def __getitem__(
        self, values: Union[int, Iterable[int], slice, PIntSet]
    ) -> Union[int, PIntSet]:
        bm: Union[PIntSet, BitMap, Exception]
        if isinstance(values, int):
            return self.bm[values]
        elif isinstance(values, (Iterable, PIntSet, BitMap)):
            bm = PIntSet()
            for index in values:
                bm.add(self.bm[int(index)])
            return bm
        elif isinstance(values, slice):
            bm = self.bm[values]
        else:
            raise ValueError("Invalid index for PIntSet")
        # Fixes a bug in BitMap
        if isinstance(bm, BitMap):
            return PIntSet(bm)
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
        Add new values from either a PIntSet, an array, a slice, or an Iterable
        """
        try:
            self.bm.update(self, values)  # type: ignore
        except TypeError:
            if values is None:
                return
            # NP check the copy here for slice
            if isinstance(values, slice):
                values = range(*values.indices(values.stop + 1))
            # do not call PIntSet constructor here cause
            # BitMap constructor calls update=>infinite recursion
            values = array.array("I", values)
            self.bm.update(self, values)  # type: ignore

    def pop(self, length: int = 1) -> PIntSet:
        "Remove one or many items and return them as a PIntSet"
        if length >= len(self):
            ret = PIntSet(self)
            self.clear()
            return ret
        ret = self[0:length]
        self -= ret
        return ret

    def remove(self, value: int) -> None:
        self.bm.remove(value)

    def remove_range(self, start: int, stop: int) -> None:
        self.bm.remove_range(start, stop)

    def symmetric_difference(self, other: PIntSet) -> PIntSet:
        return PIntSet(self.bm.symmetric_difference(other.bm))

    def symmetric_difference_update(self, other: PIntSet) -> None:
        self.bm.symmetric_difference_update(other.bm)

    def difference_update(self, *others: PIntSet) -> None:
        bms = [elt.bm for elt in others]
        self.bm.difference_update(*bms)

    def contains_range(self, start: int, end: int) -> bool:
        return self.bm.contains_range(start, end)

    @staticmethod
    def difference(*pintsets: PIntSet) -> PIntSet:
        return PIntSet(BitMap.difference(*[b.bm for b in pintsets]))

    def difference_cardinality(self, other: PIntSet) -> int:
        return self.bm.difference_cardinality(other.bm)

    def to_slice_maybe(self) -> Union[slice, PIntSet]:
        "Convert this PIntSet to a slice if possible, or return self"
        length = len(self)
        if length == 0:
            return slice(0, 0)
        first = self.min()
        last = self.max()
        if last - first + 1 == length:
            return slice(first, last + 1)
        return self

    @staticmethod
    def aspintset(x: Any) -> PIntSet:
        "Try to coerce the value as a PIntSet"
        if x is None:
            return PIntSet()
        if isinstance(x, PIntSet):
            return x
        if isinstance(x, _integer_types):
            return PIntSet([x])
        return PIntSet(x)

    def __or__(self, other: Optional[PIntSet]) -> PIntSet:
        if other is None:
            return PIntSet(self.bm)
        if isinstance(other, PIntSet):
            return PIntSet(BitMap.__or__(self.bm, other.bm))
        raise TypeError("Invalid type {type(other)}")

    def __and__(self, other: Optional[PIntSet]) -> PIntSet:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, PIntSet):
            return PIntSet(BitMap.__and__(self.bm, other.bm))
        raise TypeError("Invalid type {type(other)}")

    def __xor__(self, other: Optional[PIntSet]) -> PIntSet:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, PIntSet):
            return PIntSet(BitMap.__xor__(self.bm, other.bm))
        raise TypeError("Invalid type {type(other)}")

    def __sub__(self, other: Optional[PIntSet]) -> PIntSet:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, PIntSet):
            return PIntSet(BitMap.__sub__(self.bm, other.bm))
        raise TypeError("Invalid type {type(other)}")

    def __ior__(self, other: Optional[PIntSet]) -> PIntSet:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, PIntSet):
            BitMap.__ior__(self.bm, other.bm)
            return self
        raise TypeError("Invalid type {type(other)}")

    def __iand__(self, other: Optional[PIntSet]) -> PIntSet:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, PIntSet):
            BitMap.__iand__(self.bm, other.bm)
            return self
        raise TypeError("Invalid type {type(other)}")

    def __ixor__(self, other: Optional[PIntSet]) -> PIntSet:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, PIntSet):
            BitMap.__ixor__(self.bm, other.bm)
            return self
        raise TypeError("Invalid type {type(other)}")

    def __isub__(self, other: Optional[PIntSet]) -> PIntSet:
        if other is None:
            other = NIL_BITMAP
        if isinstance(other, PIntSet):
            BitMap.__isub__(self.bm, other.bm)
            return self
        raise TypeError("Invalid type {type(other)}")

    def flip(self, start: int, end: int) -> PIntSet:
        """
        Compute the negation of the PIntSet within the specified interval.
        """
        return PIntSet(BitMap.flip(self.bm, start, end))

    @staticmethod
    def union(*pintsets: PIntSet) -> PIntSet:
        """
        Return the union of the pintsets.
        """
        bm = BitMap.union(*[b.bm for b in pintsets])
        return PIntSet(bm) if isinstance(bm, BitMap) else bm

    @staticmethod
    def intersection(*pintsets: PIntSet) -> PIntSet:
        """
        Return the intersection of the pintsets.
        """
        bm = BitMap.intersection(*[b.bm for b in pintsets])
        return PIntSet(bm) if isinstance(bm, BitMap) else bm

    @staticmethod
    def deserialize(buff: bytes) -> PIntSet:
        """
        Generate a PIntSet from the given serialization.
        """
        return PIntSet(BitMap.deserialize(buff))

    def serialize(self) -> bytes:
        return self.bm.serialize()


NIL_BITMAP = PIntSet(FrozenBitMap())
