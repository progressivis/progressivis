from __future__ import annotations

from typing import Any as AnyType

class Unicode:
    def __init__(self, *args: AnyType, **kw: AnyType) -> None: ...
    def tag(*args: AnyType, **kw: AnyType) -> Unicode: ...

class Any:
    def __init__(self, *args: AnyType, **kw: AnyType) -> None: ...
    def tag(*args: AnyType, **kw: AnyType) -> Any: ...
