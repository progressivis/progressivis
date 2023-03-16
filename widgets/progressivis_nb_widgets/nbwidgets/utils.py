from __future__ import annotations
import time
import ipywidgets as widgets
from ipydatawidgets.ndarray.serializers import (  # type: ignore
    array_to_compressed_json,
    array_from_compressed_json,
)
from progressivis.core import asynchronize, aio


from typing import Any, Callable, Dict, Type, cast


# cf. https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Asynchronous.html
def wait_for_change(widget: Any, value: Any) -> aio.Future[Any]:
    future: aio.Future[Any] = aio.Future()

    def getvalue(change: Any) -> None:
        # make the new value available
        future.set_result(change.new)
        widget.unobserve(getvalue, value)

    widget.observe(getvalue, value)
    return future


def wait_for_click(btn: Any, cb: Callable[..., None]) -> aio.Future[Any]:
    future: aio.Future[Any] = aio.Future()

    def proc_(_: Any) -> None:
        future.set_result(True)
        btn.on_click(proc_, remove=True)
        cb()

    btn.on_click(proc_)
    return future


async def update_widget(wg: Any, attr: str, val: Any) -> None:
    await asynchronize(setattr, wg, attr, val)


class HistorizedBox(widgets.VBox):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)

    def update(self, *args: Any, **kw: Any) -> None:
        raise ValueError("Not Implemented")


def historized_widget(widget_class: Type[Any], update_method: str) -> Type[HistorizedBox]:
    from . import PrevImages

    class _Hz(HistorizedBox):
        def __init__(self, *args: Any, **kw: Any) -> None:
            self.widget = cast(widgets.DOMWidget, widget_class(*args, **kw))
            self._update_method = update_method
            self.classname = f"historized_widget-{id(self.widget)}"
            self.widget.add_class(self.classname)
            self.history = PrevImages()
            self.history.target = self.classname  # type: ignore
            super().__init__([self.widget, self.history])

        def update(self, *args: Any, **kw: Any) -> None:
            getattr(self.widget, self._update_method)(*args, **kw)
            time.sleep(0.1)
            self.history.update()
    return cast(Type[HistorizedBox], _Hz)
#
# The functions below  (data_union_to_json_compress, data_union_from_json_compress)
# are adapted from ipydatawidgets
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
#


def data_union_to_json_compress(value: Any, widget: Any) -> Dict[str, Any]:
    """Serializer for union of NDArray and NDArrayWidget"""
    if isinstance(value, widgets.DOMWidget):
        return widgets.widget_serialization["to_json"](value, widget)  # type: ignore
    return array_to_compressed_json(value, widget)  # type: ignore


def data_union_from_json_compress(value: Any, widget: Any) -> Any:
    """Deserializer for union of NDArray and NDArrayWidget"""
    if isinstance(value, str) and value.startswith("IPY_MODEL_"):
        return widgets.widget_serialization["from_json"](value, widget)  # type: ignore
    return array_from_compressed_json(value, widget)


data_union_serialization_compress: Dict[str, Callable[[Any, Any], Any]] = dict(
    to_json=data_union_to_json_compress, from_json=data_union_from_json_compress
)
