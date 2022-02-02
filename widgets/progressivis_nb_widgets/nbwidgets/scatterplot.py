from __future__ import annotations

import numpy as np
import ipywidgets as widgets
from ipydatawidgets import DataUnion
from ipydatawidgets.widgets import DataWidget
from traitlets import Unicode, Any, Bool
from progressivis.core import JSONEncoderNp as JS, asynchronize
import progressivis.core.aio as aio
from .utils import data_union_serialization_compress

from typing import Any as AnyType, List, Coroutine, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis import Module
    from progressivis.vis.mcscatterplot import MCScatterPlot
WidgetType = AnyType


# See js/lib/widgets.js for the frontend counterpart to this file.

_serialization = data_union_serialization_compress


@widgets.register
class Scatterplot(DataWidget, widgets.DOMWidget):  # type: ignore
    """Progressivis Scatterplot widget."""

    # Name of the widget view class in front-end
    _view_name = Unicode("ScatterplotView").tag(sync=True)

    # Name of the widget model class in front-end
    _model_name = Unicode("ScatterplotModel").tag(sync=True)

    # Name of the front-end module containing widget view
    _view_module = Unicode("progressivis-nb-widgets").tag(sync=True)

    # Name of the front-end module containing widget model
    _model_module = Unicode("progressivis-nb-widgets").tag(sync=True)

    # Version of the front-end module containing widget view
    _view_module_version = Unicode("^0.1.0").tag(sync=True)
    # Version of the front-end module containing widget model
    _model_module_version = Unicode("^0.1.0").tag(sync=True)

    hists = DataUnion([], dtype="int32").tag(sync=True, **_serialization)
    samples = DataUnion(np.zeros((0, 0, 0), dtype="float32"), dtype="float32").tag(
        sync=True, **_serialization
    )
    data = Unicode("{}").tag(sync=True)
    value = Any("{}").tag(sync=True)
    move_point = Any("{}").tag(sync=True)
    modal = Bool(False).tag(sync=True)
    to_hide = Any("[]").tag(sync=True)

    def link_module(
        self, module: MCScatterPlot, refresh: bool = True
    ) -> List[Coroutine[Any, Any, None]]:
        def _feed_widget(wg: WidgetType, m: MCScatterPlot) -> None:
            val = m.to_json()
            data_ = {
                k: v
                for (k, v) in val.items()
                if k not in ("hist_tensor", "sample_tensor")
            }
            ht = val.get("hist_tensor", None)
            if ht is not None:
                wg.hists = ht
            st = val.get("sample_tensor", None)
            if st is not None:
                wg.samples = st
            wg.data = JS.dumps(data_)

        async def _after_run(
            m: Module, run_number: int
        ) -> None:  # pylint: disable=unused-argument
            if not self.modal:
                await asynchronize(_feed_widget, self, m)

        module.on_after_run(_after_run)

        def from_input_value(_val: Any) -> None:
            bounds = self.value

            async def _cbk():
                await module.min_value.from_input(bounds["min"])
                await module.max_value.from_input(bounds["max"])

            aio.create_task(_cbk())

        self.observe(from_input_value, "value")

        def from_input_move_point(_val: Any) -> None:
            aio.create_task(module.move_point.from_input(self.move_point))

        self.observe(from_input_move_point, "move_point")

        def awake(_val: Any) -> []:
            if module._json_cache is None or self.modal:
                return
            dummy = module._json_cache.get("dummy", 555)
            module._json_cache["dummy"] = -dummy
            aio.create_task(asynchronize(_feed_widget, self, module))  # TODO: improve

        self.observe(awake, "modal")
        return []

    def __init__(self, *, disable: Sequence[Any] = tuple()):
        super().__init__()
        self.to_hide = list(disable)
