from __future__ import annotations

import numpy as np

from typing import Any, Optional, Dict, Union, Literal


def evaluate(
    v: str,
    local_dict: Optional[Dict[str, Any]] = None,
    global_dict: Optional[Dict[str, Any]] = None,
    out: Optional[np.ndarray[Any, Any]] = None,
    order: Union[Literal["C"],
                 Literal["F"],
                 Literal["A"],
                 Literal["K"]] = "K",
    casting: Union[Literal["safe"],
                   Literal["no"],
                   Literal["equiv"],
                   Literal["same_kind"],
                   Literal["unsafe"]] = "safe",
    **kwargs: Any
) -> np.ndarray[Any, Any]: ...
