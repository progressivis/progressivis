import ipywidgets as widgets  # type: ignore
from traitlets import Unicode  # type: ignore

# See js/lib/widgets.js for the frontend counterpart to this file.


@widgets.register
class ModuleGraph(widgets.DOMWidget):
    """Progressivis ModuleGraph widget."""

    # Name of the widget view class in front-end
    _view_name = Unicode("ModuleGraphView").tag(sync=True)

    # Name of the widget model class in front-end
    _model_name = Unicode("ModuleGraphModel").tag(sync=True)

    # Name of the front-end module containing widget view
    _view_module = Unicode("progressivis-nb-widgets").tag(sync=True)

    # Name of the front-end module containing widget model
    _model_module = Unicode("progressivis-nb-widgets").tag(sync=True)

    # Version of the front-end module containing widget view
    _view_module_version = Unicode("^0.1.0").tag(sync=True)
    # Version of the front-end module containing widget model
    _model_module_version = Unicode("^0.1.0").tag(sync=True)

    data = Unicode("{}").tag(sync=True)
