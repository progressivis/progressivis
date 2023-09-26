import numpy as np
from progressivis.linalg import unary_module, binary_module, reduce_module
from typing import cast


@unary_module
def CustomUnary(x: float) -> float:
    return cast(float, (x + np.sin(x)) / (x + np.cos(x)))


@binary_module
def CustomBinary(x: float, y: float) -> float:
    return cast(float, (x + np.sin(y)) / (x + np.cos(y)))


@reduce_module
def CustomBinaryReduce(x: float, y: float) -> float:
    return cast(float, (x + np.sin(y)) / (x + np.cos(y)))
