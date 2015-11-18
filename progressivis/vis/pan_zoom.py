from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class PanZoom(DataFrameModule):
    parameters = [('history',np.dtype(int),   3) ]
    
    schema = [('xmin',   np.dtype(float), np.nan),
              ('xmax',   np.dtype(float), np.nan),
              ('ymin',   np.dtype(float), np.nan),
              ('ymax',   np.dtype(float), np.nan),
              DataFrameModule.UPDATE_COLUMN_DESC]

    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('bounds', type=pd.DataFrame, required=True),
                         SlotDescriptor('viewport', type=pd.DataFrame, required=False)])
        super(PanZoom, self).__init__(dataframe_slot='panzoom', **kwds)
        self._df = self.create_dataframe(PanZoom.schema)

    def predict_step_size(self, duration):
        return 1

    def _add_input_slot(self, name):
        self.inputs.append(name)
        self.input_descriptors[name] = SlotDescriptor(name, type=pd.DataFrame, required=True)
        self._input_slots[name] = None

    # Magic input slot created 
    def _connect_input(self, slot):
        ret = self.get_input_slot(slot.input_name)
        if ret and slot.input_name=='viewport':
            name = 'viewport.%d' % (len(self.input)-1)
            self._add_input_slot(name)
            slot.input_name = name # patch the slot name
            ret = None
        self._input_slots[slot.input_name] = slot
        return ret
    
    def run_step(self,run_number,step_size,howlong):
        bounds = self.last_row(self.get_input_slot('bounds').data())
        
        if bounds is None:
            return self._return_run_step(self.state_blocked, steps_run=1)
        (xmin, xmax, ymin, ymax) = bounds[['xmin', 'xmax', 'ymin', 'ymax']]
        for name in self._input_slots:
            if not name.startswith('viewport'): continue
            vp = self.get_input_slot(name)
            if vp is None or vp.data() is None: continue
            vp = vp.data()
            (x1,x2,y1,y2)=self.last_row(vp)[['xmin', 'xmax', 'ymin', 'ymax']]
            if x1 > xmin: xmin = np.min([x1, xmax])
            if x2 < xmax: xmax = np.max([x2, xmin])
            if y1 > ymin: ymin = np.min([y1, ymax])
            if y2 < ymax: ymax = np.max([y2, ymin])
        self._df.loc[run_number] = {
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            DataFrameModule.UPDATE_COLUMN: run_number }
        return self._return_run_step(self.state_blocked, steps_run=1)
