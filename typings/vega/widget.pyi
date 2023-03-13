from __future__ import annotations

from typing import Any, Dict, Tuple, Type, Sequence, Union, List, Callable


class VegaWidget:
    value: Any
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable[..., None], attr: str) -> None: ...
