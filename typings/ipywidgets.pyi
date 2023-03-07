from __future__ import annotations

from typing import Any, Dict, Tuple, Type, Sequence, Union, List, Callable

def register(*args: Any) -> Any: ...

class Widget:
    description: str
    value: Any
    def observe(self, cb: Callable, names: str) -> None: ...

class IntSlider(Widget):
    value: Any
    def __init__(self, **kw: Any) -> None: ...


class IntRangeSlider(Widget):
    value: Any
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable, attr: str) -> None: ...

class Button(Widget):
    disabled: bool
    def __init__(self, **kw: Any) -> None: ...
    def on_click(self, cb: Callable) -> None: ...


class SelectBase(Widget):
    disabled: bool
    options: List[str]
    value: List[str]
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable, names: str) -> None: ...

class SelectMultiple(SelectBase):
    ...

class Select(SelectBase):
    ...

class Dropdown(SelectBase):
    ...

class Label(Widget):
    value: str
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Checkbox(Widget):
    disabled: bool
    value: bool
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable, names: str) -> None: ...

class Layout(Widget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Box(Widget):
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


class Output(Widget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Text(Widget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...


class IntText(Widget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...


class Textarea(Widget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class RadioButtons(Widget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...


class HTML(Widget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...
