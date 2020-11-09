"""Manage bit sets as ordered list of integers very efficiently, relying on RoaringBitmaps.
"""

import array

from pyroaring import BitMap
import numpy as np
from  collections.abc import Iterable
# pragma no cover
# pylint: disable=invalid-name
_integer_types = (int, np.integer)


class bitmap(BitMap,object):
    # pylint: disable=invalid-name
    """
    Derive from an efficient and light-weight ordered set of 32 bits integers.
    """
    def __new__(cls, values=None, copy_on_write=False, optimize=True, no_init=False):
        if isinstance(values, slice):
            values = range(values.start, values.stop,
                           (values.step if values.step else 1))
        return super(bitmap, cls).__new__(cls, values,
                                          copy_on_write,
                                          optimize,
                                          no_init)
        #BitMap.__init__(self, values, copy_on_write)

    def clear(self):
        "Clear the bitmap in-place"
        self &= NIL_BITMAP

    def __contains__(self, other):
        if isinstance(other, _integer_types):
            return BitMap.__contains__(self, other)
        other = self.asbitmap(other)
        return other <= self

    def __repr__(self):
        l = len(self)
        if l > 10:
            values = ', '.join([str(n) for n in self[0:6]])
            values += '...(%d)...%d)' % (l, self[l-1])
        else:
            values = ', '.join([str(n) for n in self])
        return 'bitmap([%s])' % values

    # def __binary_op__(self, other, function):
    #     if other is None:
    #         other = NIL_BITMAP
    #     try:
    #         return bitmap(obj=function(self.__obj__, other.__obj__))
    #     except AttributeError:
    #         raise TypeError('Not a bitmap.')

    # def __binary_op_inplace__(self, other, function):
    #     if other is None:
    #         other = NIL_BITMAP
    #     return BitMap.__binary_op_inplace__(self, other, function)

    def __getitem__(self, values):
        bm = BitMap.__getitem__(self, values)
        if isinstance(bm, BitMap):
            return bitmap(bm)
        if (isinstance(bm, TypeError) and isinstance(values, Iterable)):
            bmval = bitmap(values)
            if bmval in self:
                return bmval
        if isinstance(bm, Exception):
            raise bm
        raise KeyError(f"wrong key{values}")

                
    # def __eq__(self, other):
    #     if isinstance(other, BitMap):
    #         return BitMap.__eq__(other, self)
    #     return BitMap.__eq__(self, other)

    def update(self, values):
        '''
        Add new values from either a bitmap, an array, a slice, or an Iterable
        '''
        try:
            BitMap.update(self, values)
        except TypeError:
            if values is None:
                return
            # NP check the copy here for slice
            if isinstance(values, slice):
                values = range(*values.indices(values.stop+1))
            # do not call bitmap constructor here cause
            # BitMap constructor calls update=>infinite recursion
            values = array.array('I', values)
            BitMap.update(self, values)

    def pop(self, length=1):
        "Remove one or many items and return them as a bitmap"
        if length >= len(self):
            ret = bitmap(self)
            self &= NIL_BITMAP
            return ret
        ret = self[0:length]
        self -= ret
        return bitmap(ret)

    def to_slice_maybe(self):
        "Convert this bitmap to a slice if possible, or return self"
        length = len(self)
        if length == 0:
            return slice(0, 0)
        first = self.min()
        last = self.max()
        if last-first+1 == length:
            return slice(first, last+1)
        return self

    @staticmethod
    def asbitmap(x):
        "Try to coerce the value as a bitmap"
        if x is None:
            return NIL_BITMAP
        if isinstance(x, bitmap):
            return x
        if isinstance(x, _integer_types):
            return bitmap([x])
        return bitmap(x)

    def __or__(self, other):
        if other is None:
            other = NIL_BITMAP
        return bitmap(BitMap.__or__(self, other))

    def __and__(self, other):
        if other is None:
            other = NIL_BITMAP
        return bitmap(BitMap.__and__(self, other))

    def __xor__(self, other):
        if other is None:
            other = NIL_BITMAP
        return bitmap(BitMap.__xor__(self, other))

    def __sub__(self, other):
        if other is None:
            other = NIL_BITMAP
        return bitmap(BitMap.__sub__(self, other))

    def __ior__(self, other):
        if other is None:
            other = NIL_BITMAP
        return BitMap.__ior__(self, other)

    def __iand__(self, other):
        if other is None:
            other = NIL_BITMAP
        return BitMap.__iand__(self, other)

    def __ixor__(self, other):
        if other is None:
            other = NIL_BITMAP
        return BitMap.__ixor__(self, other)

    def __isub__(self, other):
        if other is None:
            other = NIL_BITMAP
        return BitMap.__isub__(self, other)

    def flip(self, start, end):
        """
        Compute the negation of the bitmap within the specified interval.
        """
        return bitmap(BitMap.flip(self, start, end))

    @staticmethod
    def union(*bitmaps):
        """
        Return the union of the bitmaps.
        """
        bm = BitMap.union(*bitmaps)
        return bitmap(bm) if isinstance(bm, BitMap) else bm

    @staticmethod
    def intersection(*bitmaps):
        """
        Return the intersection of the bitmaps.
        """
        bm = BitMap.intersection(*bitmaps)
        return bitmap(bm) if isinstance(bm, BitMap) else bm

    @staticmethod
    def deserialize(buff):
        """
        Generate a bitmap from the given serialization.
        """
        return bitmap(BitMap.deserialize(buff))

NIL_BITMAP = bitmap()
