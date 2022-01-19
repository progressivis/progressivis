from __future__ import annotations

import numpy as np
import ipywidgets as widgets  # type: ignore
from ipydatawidgets import DataUnion  # type: ignore
from ipydatawidgets.widgets import DataWidget  # type: ignore
from traitlets import Unicode, Any, Bool  # type: ignore
from progressivis.core import JSONEncoderNp as JS, asynchronize
import progressivis.core.aio as aio

from .utils import data_union_serialization_compress, wait_for_change

from typing import Any as AnyType, List, Coroutine, NoReturn, Sequence, TYPE_CHECKING

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
        self,
        module: MCScatterPlot,
        refresh: bool = True
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


        async def _refresh() -> NoReturn:
            while True:
                await aio.sleep(0.5)

        async def _after_run(
            m: Module,
            run_number: int
        ) -> None:  # pylint: disable=unused-argument
            if not self.modal:
                await asynchronize(_feed_widget, self, m)

        module.on_after_run(_after_run)


        async def _from_input_value() -> None:
            while True:
                await wait_for_change(self, "value")
                bounds = self.value
                min_value = module.min_value
                max_value = module.max_value
                assert min_value is not None and max_value is not None
                await min_value.from_input(bounds["min"])
                await max_value.from_input(bounds["max"])

        async def _from_input_move_point() -> None:
            while True:
                await wait_for_change(self, "move_point")
                print(f"Should move point to {self.move_point}")
                # await module.move_point.from_input(self.move_point)

        async def _awake() -> None:
            """
            Hack intended to force the rendering even if the data
            are exhausted at the time of the first display
            """
            while True:
                await wait_for_change(self, "modal")
                # pylint: disable=protected-access
                if module._json_cache is None or self.modal:
                    continue
                dummy = module._json_cache.get("dummy", 555)
                module._json_cache["dummy"] = -dummy
                await asynchronize(_feed_widget, self, module)
        return [
            _from_input_value(),
            _from_input_move_point(),
            _awake(),
        ] + ([_refresh()] if refresh else [])

    def __init__(self, *, disable: Sequence[Any] = tuple()):
        super().__init__()
        self.to_hide = list(disable)
