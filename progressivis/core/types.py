from __future__ import annotations

import numpy as np

from typing import Any, Dict, Union, Tuple, Sequence

JSon = Dict[str, Any]
Index = Any  # simplify for now
Chunks = Union[None, int, Tuple[int, ...], Dict[str, Union[int, Tuple[int, ...]]]]
Shape = Sequence[int]
Indexer = Any  # improve later
ColIndexer = Union[int, np.integer, str]
