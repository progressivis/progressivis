from __future__ import absolute_import, division, print_function

from collections import namedtuple
import pandas as pd

NIL_INDEX = pd.Index([])

IndexDiff = namedtuple('IndexDiff', 'created,kept,deleted')

if pd.__version__ > '0.18':
    def IndexDifference(index1, index2):
        if not isinstance(index1, pd.RangeIndex) \
          or not isinstance(index2, pd.RangeIndex) \
          or index1._step != 1 or index2._step != 1:
            return index1.difference(index2)
        if index1._stop < index2._start or index2._stop < index1._start:
            return index1
        if index1._start <= index2._start:
            if index1._stop <= index2._stop:
                return pd.RangeIndex(index1._start, index2._start)
            return index1.difference(index2)
        if index1._stop >= index2._stop:
            return pd.RangeIndex(index2._stop, index1._stop)
        return NIL_INDEX
else:
    def IndexDifference(index1, index2):
        return index1.difference(index2)


def index_changes(index1, index2):
    if index1.equals(index2):
        return IndexDiff(created=NIL_INDEX, kept=index1, deleted=NIL_INDEX)
    created = IndexDifference(index2, index1)
    kept = index2.intersection(index1)
    deleted = IndexDifference(index1, index2)
    return IndexDiff(created=created, kept=kept, deleted=deleted)
