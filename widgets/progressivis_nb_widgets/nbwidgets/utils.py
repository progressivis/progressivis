from __future__ import annotations

import ipywidgets as widgets  # type: ignore
from ipydatawidgets.ndarray.serializers import (  # type: ignore
    array_to_compressed_json,
    array_from_compressed_json,
)
from progressivis.core import asynchronize, aio

from typing import Any, Callable, Dict


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


#
# The functions below  (data_union_to_json_compress, data_union_from_json_compress)
# are adapted from ipydatawidgets
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
#


def data_union_to_json_compress(value: Any, widget: Any) -> Dict[str, Any]:
    """Serializer for union of NDArray and NDArrayWidget"""
    if isinstance(value, widgets.Widget):
        return widgets.widget_serialization["to_json"](value, widget)  # type: ignore
    return array_to_compressed_json(value, widget)  # type: ignore


def data_union_from_json_compress(value: Any, widget: Any) -> Any:
    """Deserializer for union of NDArray and NDArrayWidget"""
    if isinstance(value, str) and value.startswith("IPY_MODEL_"):
        return widgets.widget_serialization["from_json"](value, widget)
    return array_from_compressed_json(value, widget)


data_union_serialization_compress: Dict[str, Callable[[Any, Any], Any]] = dict(
    to_json=data_union_to_json_compress, from_json=data_union_from_json_compress
)
