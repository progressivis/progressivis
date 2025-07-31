from __future__ import annotations

from typing import Any, Sequence, List, Callable

def register(*args: Any) -> Any: ...

class DOMWidget:
    description: str
    value: Any
    layout: Any
    disabled: bool
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable[..., None], names: str) -> None: ...
    def send(self, *args: Any, **kw: Any) -> None: ...
    def add_class(self, *args: Any, **kw: Any) -> None: ...

class IntSlider(DOMWidget):
    value: int
    min: int
    max: int
    step: int
    def __init__(self, **kw: Any) -> None: ...

class IntProgress(DOMWidget):
    value: int
    min: int
    max: int
    def __init__(self, **kw: Any) -> None: ...

class FloatProgress(DOMWidget):
    value: float
    def __init__(self, **kw: Any) -> None: ...

class FileUpload(DOMWidget):
    def __init__(self, **kw: Any) -> None: ...

class IntRangeSlider(DOMWidget):
    value: Any
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable[..., None], attr: str) -> None: ...

class BoundedIntText(DOMWidget):
    value: Any
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable[..., None], attr: str) -> None: ...

class BoundedFloatText(DOMWidget):
    value: Any
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable[..., None], attr: str) -> None: ...

class Button(DOMWidget):
    def __init__(self, **kw: Any) -> None: ...
    def on_click(self, cb: Callable[..., None]) -> None: ...


class SelectBase(DOMWidget):
    options: List[str]
    value: Any
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable[..., None], names: str) -> None: ...

class SelectMultiple(SelectBase):
    value: List[str]


class Select(SelectBase):
    value: str


class Combobox(SelectBase):
    ...

class Dropdown(SelectBase):
    ...


class Label(DOMWidget):
    value: str
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Checkbox(DOMWidget):
    value: bool
    def __init__(self, **kw: Any) -> None: ...
    def observe(self, cb: Callable[..., None], names: str) -> None: ...

class Layout(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Box(DOMWidget):
    children: Sequence[DOMWidget]
    display: Any
    def __init__(self, *args: Any, **kw: Any) -> None: ...


class VBox(Box):
    ...

class HBox(Box):
    ...

class GridBox(Box):
    ...

class Tab(Box):
    selected_index: int
    titles: Sequence[str]
    def set_title(self, index: int, name: str) -> None: ...
    def get_title(self, index: int) -> str: ...

class Accordion(Tab): ...

class Output(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Text(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...


class IntText(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class FloatText(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class Textarea(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...

class RadioButtons(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...


class HTML(DOMWidget):
    def __init__(self, *args: Any, **kw: Any) -> None: ...
