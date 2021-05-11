import ipywidgets as widgets
from ipydatawidgets.ndarray.serializers import (
    array_to_compressed_json,
    array_from_compressed_json)
from progressivis.core import asynchronize, aio

async def update_widget(wg, attr, val):
    await asynchronize(setattr, wg, attr, val)

#
# The functions below  (data_union_to_json_compress, data_union_from_json_compress)
# are adapted from ipydatawidgets
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
#


def data_union_to_json_compress(value, widget):
    """Serializer for union of NDArray and NDArrayWidget"""
    if isinstance(value, widgets.Widget):
        return widgets.widget_serialization['to_json'](value, widget)
    return array_to_compressed_json(value, widget)


def data_union_from_json_compress(value, widget):
    """Deserializer for union of NDArray and NDArrayWidget"""
    if isinstance(value, str) and value.startswith('IPY_MODEL_'):
        return widgets.widget_serialization['from_json'](value, widget)
    return array_from_compressed_json(value, widget)


data_union_serialization_compress = dict(
    to_json=data_union_to_json_compress,
    from_json=data_union_from_json_compress)
