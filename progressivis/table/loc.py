from __future__ import annotations

import numpy as np

from progressivis.core.utils import integer_types
from progressivis.core.pintset import PIntSet

from typing import Any, Iterable


class Loc:
    ITERABLE = 8
    INT = 1
    LIST = 2 | ITERABLE
    SLICE = 3
    NDARRAY = 4 | ITERABLE
    NDBOOLARRAY = 6
    BITMAP = 5 | ITERABLE

    @staticmethod
    def isiterable(loc: int) -> bool:
        return (loc & Loc.ITERABLE) != 0

    @staticmethod
    def dispatch(locs: Any) -> int:
        if isinstance(locs, integer_types):
            return Loc.INT
        elif isinstance(locs, slice):
            return Loc.SLICE
        elif isinstance(locs, np.ndarray):
            if locs.dtype == np.int32 or locs.dtype == np.int64:
                return Loc.NDARRAY
            elif locs.dtype == np.bool_:
                return Loc.NDBOOLARRAY
        elif isinstance(locs, PIntSet):
            return Loc.BITMAP
        elif isinstance(locs, Iterable):
            return Loc.ITERABLE
        raise ValueError("Unhandled type for %s", locs)

    @staticmethod
    def to_iterable(locs: Any, size: int) -> Iterable[Any]:
        loc = Loc.dispatch(locs)

        if Loc.isiterable(loc):
            assert isinstance(locs, Iterable)
            return locs
        elif loc == Loc.INT:
            return [locs]
        elif loc == Loc.SLICE:
            return range(*locs.index(size))
        elif loc == Loc.NDARRAY:
            assert isinstance(locs, Iterable)
            return locs
        elif loc == Loc.NDBOOLARRAY:
            return np.where(locs)[0]
        raise ValueError("Cannot convert %s into an iterable", locs)
