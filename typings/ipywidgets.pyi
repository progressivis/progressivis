from __future__ import annotations

from typing import Any, Dict, Tuple, Type, Sequence, Union, List, Callable

def register(*args: Any) -> Any: ...

class DOMWidget:
    description: str
    value: Any
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable[..., None], names: str) -> None: ...
    def send(self, *args: Any, **kw: Any) -> None: ...


class IntSlider(DOMWidget):
    value: int
    def __init__(self, **kw: Any) -> None: ...


class IntRangeSlider(DOMWidget):
    value: Any
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable[..., None], attr: str) -> None: ...

class Button(DOMWidget):
    disabled: bool
    def __init__(self, **kw: Any) -> None: ...
    def on_click(self, cb: Callable[..., None]) -> None: ...


class SelectBase(DOMWidget):
    disabled: bool
    options: List[str]
    value: Any
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable[..., None], names: str) -> None: ...

class SelectMultiple(SelectBase):
    value: List[str]


class Select(SelectBase):
    value: str


class Dropdown(SelectBase):
    ...

class Label(DOMWidget):
    value: str
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Checkbox(DOMWidget):
    disabled: bool
    value: bool
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable[..., None], names: str) -> None: ...

class Layout(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Box(DOMWidget):
    disabled: bool
    children: Union[Tuple[Any, ...], List[Any]]
    def __init__(self, *args: Any, **kw: Any) -> None: ...


class VBox(Box):
    ...

class HBox(Box):
    ...

class GridBox(Box):
    ...

class Tab(Box):
    selected_index: int
    def set_title(self, index: int, name: str) -> None: ...
    def get_title(self, index: int) -> str: ...


class Output(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Text(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...


class IntText(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...


class Textarea(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class RadioButtons(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...


class HTML(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...
