from __future__ import absolute_import, division, print_function

from .nary import NAry
from .table import Table


class Merge(NAry):
    def __init__(self, **kwds):
        """Merge(how='inner', on=None, left_on=None, right_on=None,left_index=False, right_index=False, sort=False,suffixes=('_x', '_y'), copy=True, indicator=False,id=None,tracer=None,predictor=None,storage=None,input_descriptors=[],output_descriptors=[])
        """
        super(Merge, self).__init__(**kwds)
        self.merge_kwds = self._filter_kwds(kwds, Table.merge)
        self._context = {}
    def run_step(self,run_number,step_size,howlong):
        frames = []
        for name in self.inputs:
            if not name.startswith('table'):
                continue
            slot = self.get_input_slot(name)
            with slot.lock:
                df = slot.data()
            frames.append(df)
        df = frames[0]
        for other in frames[1:]:
            if not self._context:
                df = df.merge(other, merge_ctx=self._context, **self.merge_kwds)
            else:
                df = df.merge_cont(other, merge_ctx=self._context)
        l = len(df)
        self._table = df
        return self._return_run_step(self.state_blocked, steps_run=l)
