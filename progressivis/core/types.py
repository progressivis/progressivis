from __future__ import annotations

import numpy as np

from typing import Any, Dict, Union, Tuple

JSon = Dict[str, Any]
Index = Any  # simplify for now
Chunks = Union[None, int, Dict[str, Union[int, Tuple[int, ...]]]]
Indexer = Union[Any]  # improve later
ColIndexer = Union[int, np.integer, str]
