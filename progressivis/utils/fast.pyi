import numpy as np
import numpy.typing as npt

from typing import Any


def check_contiguity(f: npt.NDArray[np.uint32], init: int = 0) -> int: ...


def indices_to_slice(indices: Any) -> slice: ...


def next_pow2(v: int) -> int: ...
