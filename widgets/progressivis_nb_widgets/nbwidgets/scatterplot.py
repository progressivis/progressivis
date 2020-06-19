import ipywidgets as widgets
from traitlets import Unicode, Any
from progressivis.core import JSONEncoderNp as JS, asynchronize
import progressivis.core.aio as aio
from .utils import *
import numpy as np
from ipydatawidgets import NDArray, shape_constraints, array_serialization, DataUnion, data_union_serialization
from ipydatawidgets.widgets import DataWidget
# See js/lib/widgets.js for the frontend counterpart to this file.

@widgets.register
class Scatterplot(DataWidget, widgets.DOMWidget):
    """Progressivis Scatterplot widget."""

    # Name of the widget view class in front-end
    _view_name = Unicode('ScatterplotView').tag(sync=True)

    # Name of the widget model class in front-end
    _model_name = Unicode('ScatterplotModel').tag(sync=True)

    # Name of the front-end module containing widget view
    _view_module = Unicode('progressivis-nb-widgets').tag(sync=True)

    # Name of the front-end module containing widget model
    _model_module = Unicode('progressivis-nb-widgets').tag(sync=True)

    # Version of the front-end module containing widget view
    _view_module_version = Unicode('^0.1.0').tag(sync=True)
    # Version of the front-end module containing widget model
    _model_module_version = Unicode('^0.1.0').tag(sync=True)

    # Widget specific property.
    # Widget properties are defined as traitlets. Any property tagged with `sync=True`
    # is automatically synced to the frontend *any* time it changes in Python.
    # It is synced back to Python from the frontend *any* time the model is touched.
    hists = DataUnion(
        [],
        dtype='int32',
        #shape_constraint=shape_constraints(None, None),  
    ).tag(sync=True, **data_union_serialization)
    samples = DataUnion(
        [],
        dtype='float32',
        #shape_constraint=shape_constraints(None, None),  
    ).tag(sync=True, **data_union_serialization)
    data = Unicode('{}').tag(sync=True)
    value =  Any('{}').tag(sync=True)
    move_point =  Any('{}').tag(sync=True)    

    def link_module(self, module):
        def _feed_widget(wg, val):
            wg.hists = val.pop('hist_tensor')
            st = val.pop('sample_tensor')
            if st is not None:
                wg.samples = st
            wg.data = JS.dumps(val)

        async def _after_run(m, run_number):
            await asynchronize(_feed_widget, self, m._json_cache)
        module.after_run_proc = _after_run
        async def _from_input_value():
            while True:
                await wait_for_change(self, 'value')
                bounds = self.value
                await module.min_value.from_input(bounds['min'])
                await module.max_value.from_input(bounds['max'])
        async def _from_input_move_point():
            while True:
                await wait_for_change(self, 'move_point')
                await module.move_point.from_input(self.move_point)
        async def _awake():
            """
            Hack intended to force the rendering even if the data 
            are exhausted at the time of the first display
            """
            while True:
                await aio.sleep(5)
                dummy = module._json_cache.get('dummy', 555)
                module._json_cache['dummy'] = -dummy
                await asynchronize(_feed_widget, self, module._json_cache)

        return [_from_input_value(), _from_input_move_point(), _awake()]
