from __future__ import annotations

from typing import Any, Dict, Tuple, Type, Sequence, Union, List, Callable

class IntRangeSlider:
    value: Any
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable, attr: str) -> None: ...

class Button:
    disabled: bool
    def __init__(self, **kw: Any) -> None: ...
    def on_click(self, cb: Callable) -> None: ...

class Tab:
    children: Tuple[Any, ...]
    def __init__(self, **kw: Any) -> None: ...
    def set_title(self, index: int, name: str) -> None: ...
    def get_title(self, index: int) -> str: ...
    def observe(self, cb: Callable, names: str) -> None: ...

class SelectMultiple:
    disabled: bool
    options: List[str]
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable, attr: str) -> None: ...

class Label:
    value: str
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Checkbox:
    disabled: bool
    value: bool
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable, attr: str) -> None: ...

class GridBox:
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Layout:
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class VBox:
    children: Tuple[Any, ...]
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class HBox:
    children: Tuple[Any, ...]
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Output:
    def __init__(self, *args: Any, **kw: Any) -> None: ...
