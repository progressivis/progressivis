import numpy as np
from collections import Iterable

from progressivis.core.utils import integer_types
from progressivis.core.bitmap import bitmap


class Loc(object):
    ITERABLE = 8
    INT = 1
    LIST = 2 | ITERABLE
    SLICE = 3
    NDARRAY = 4 | ITERABLE
    NDBOOLARRAY = 6
    BITMAP = 5 | ITERABLE

    @staticmethod
    def isiterable(loc):
        return (loc & Loc.ITERABLE) != 0

    @staticmethod
    def dispatch(locs):
        if isinstance(locs, integer_types):
            return Loc.INT
        elif isinstance(locs, slice):
            return Loc.SLICE
        elif isinstance(locs, np.ndarray):
            if locs.dtype == np.int32 or locs.dtype == np.int64:
                return Loc.NDARRAY
            elif locs.dtype == np.bool:
                return Loc.NDBOOLARRAY
        elif isinstance(locs, bitmap):
            return Loc.BITMAP
        elif isinstance(locs, Iterable):
            return Loc.ITERABLE
        raise ValueError("Unhandled type for %s", locs)

    @staticmethod
    def to_iterable(locs, size):
        loc = Loc.dispatch(locs)

        if Loc.isiterable(loc):
            return loc
        elif loc == Loc.INT:
            return [locs]
        elif loc == Loc.SLICE:
            return range(*locs.index(size))
        elif loc == Loc.NDARRAY:
            return locs
        elif loc == Loc.NDBOOLARRAY:
            return np.where(locs)[0]
        raise ValueError("Cannot convert %s into an iterable", locs)
