from ..core.bitmap import bitmap

import numpy as np
import numpy.typing as npt
from typing import Any, Iterator, Union

PROP_START_AT_0: int
PROP_MONOTONIC_INC: int
PROP_CONTIGUOUS: int
PROP_IDENTITY: int

def check_contiguity(f: npt.NDArray[np.uint32], init: int = 0) -> int: ...
def indices_to_slice(indices: Any) -> Union[slice, bitmap]: ...
def next_pow2(v: int) -> int: ...
def indices_to_slice_iterator(indices: Any) -> Iterator[slice]: ...
