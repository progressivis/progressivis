from __future__ import annotations

import ipywidgets as widgets
from traitlets import Unicode  # type: ignore

# from typing import Any as AnyType

# WidgetType = AnyType


# See js/src/previmages.js for the frontend counterpart to this file.


@widgets.register  # type: ignore
class PrevImages(widgets.DOMWidget):  # type: ignore
    """Progressivis PrevImages widget."""

    # Name of the widget view class in front-end
    _view_name = Unicode("PrevImagesView").tag(sync=True)

    # Name of the widget model class in front-end
    _model_name = Unicode("PrevImagesModel").tag(sync=True)

    # Name of the front-end module containing widget view
    _view_module = Unicode("progressivis-nb-widgets").tag(sync=True)

    # Name of the front-end module containing widget model
    _model_module = Unicode("progressivis-nb-widgets").tag(sync=True)

    # Version of the front-end module containing widget view
    _view_module_version = Unicode("^0.1.0").tag(sync=True)
    # Version of the front-end module containing widget model
    _model_module_version = Unicode("^0.1.0").tag(sync=True)

    target = Unicode("").tag(sync=True)

    def update(self) -> None:
        self.send(dict(response=42))
