from __future__ import absolute_import, division, print_function

from pyroaring import BitMap
import array
import six
import numpy as np

if six.PY2:
    range = xrange
    integer_types = (six.integer_types, np.integer)
else:
    integer_types = (int, np.integer)


class bitmap(BitMap, object):
    def __init__(self, values=None, obj=None):
        if isinstance(values, slice):
            values = range(values.start, values.stop, (values.step or 1))
        BitMap.__init__(self, values, obj)

    def clear(self):
        self &= NIL_BITMAP

    def __contains__(self, other):
        if isinstance(other, integer_types):
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

    def __binary_op__(self, other, function):
        if other is None:
            other = NIL_BITMAP
        try:
            return bitmap(obj=function(self.__obj__, other.__obj__))
        except AttributeError:
            raise TypeError('Not a bitmap.')

    def __binary_op_inplace__(self, other, function):
        if other is None:
            other = NIL_BITMAP
        return BitMap.__binary_op_inplace__(self, other, function)

    def __getitem__(self, values):
        bm = BitMap.__getitem__(self, values)
        return bitmap(bm) if isinstance(bm, BitMap) else bm

    def __eq__(self, other):
        if isinstance(other, BitMap):
            return BitMap.__eq__(other, self)
        return BitMap.__eq__(self, other)

    def update(self, values):
        if values is None:
            return
        # NP check the copy here for slice
        if not isinstance(values, (bitmap, BitMap, array.array)):
            if isinstance(values, slice):
                values = range(*values.indices(values.stop+1))
            # do not call bitmap constructor here cause
            # BitMap constructor calls update=>infinite recursion
            values = array.array('I', values)
        BitMap.update(self, values)

    def pop(self, length=1):
        l = len(self)

        if length >= l:
            ret = bitmap(self)
            self &= bitmap()
            return ret
        ret = self[0:length]
        self -= ret
        return bitmap(ret)

    def to_slice_maybe(self):
        l = len(self)
        if l == 0:
            return slice(0, 0)
        first = self.min()
        last = self.max()
        if last-first+1 == l:
            return slice(first, last+1)
        else:
            return self

    @staticmethod
    def asbitmap(x):
        if x is None:
            return NIL_BITMAP
        if isinstance(x, (bitmap, BitMap)):
            return x
        return bitmap(x)

NIL_BITMAP = bitmap()
