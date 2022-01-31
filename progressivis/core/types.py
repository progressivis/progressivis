from __future__ import annotations

import numpy as np

from typing import Any, List, Dict, Union, Tuple, Sequence, TypeVar, Optional


JSon = Dict[str, Any]
Index = Any  # simplify for now
Chunks = Union[None, int, Tuple[int, ...], Dict[str, Union[int, Tuple[int, ...]]]]
Shape = Sequence[int]
Indexer = Any  # improve later
ColIndexer = Union[int, np.integer, str]
R = TypeVar("R")
Floats = Union[np.ndarray[Any, Any], List[float]]


def notNone(x: Optional[R]) -> R:
    assert x is not None
    return x
