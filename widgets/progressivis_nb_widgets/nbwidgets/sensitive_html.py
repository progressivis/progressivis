import ipywidgets as widgets
from traitlets import Unicode, Dict, Any

# See js/lib/widgets.js for the frontend counterpart to this file.


@widgets.register
class SensitiveHTML(widgets.DOMWidget):
    """Progressivis SensitiveHTML widget."""

    # Name of the widget view class in front-end
    _view_name = Unicode('SensitiveHTMLView').tag(sync=True)

    # Name of the widget model class in front-end
    _model_name = Unicode('SensitiveHTMLModel').tag(sync=True)

    # Name of the front-end module containing widget view
    _view_module = Unicode('progressivis-nb-widgets').tag(sync=True)

    # Name of the front-end module containing widget model
    _model_module = Unicode('progressivis-nb-widgets').tag(sync=True)

    # Version of the front-end module containing widget view
    _view_module_version = Unicode('^0.1.0').tag(sync=True)
    # Version of the front-end module containing widget model
    _model_module_version = Unicode('^0.1.0').tag(sync=True)

    html = Unicode('').tag(sync=True)
    data = Any({}).tag(sync=True)
    value = Any({}).tag(sync=True)
    sensitive_css_class = Unicode('aCssClass').tag(sync=True)
