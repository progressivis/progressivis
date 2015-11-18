from .nary import NAry

import pandas as pd

class Merge(NAry):
    def __init__(self, **kwds):
        """Merge(how='inner', on=None, left_on=None, right_on=None,left_index=False, right_index=False, sort=False,suffixes=('_x', '_y'), copy=True, indicator=False,id=None,scheduler=None,tracer=None,predictor=None,storage=None,input_descriptors=[],output_descriptors=[])
        """
        super(Merge, self).__init__(**kwds)
        self.merge_kwds = self._filter_kwds(kwds, pd.merge)
        
    def run_step(self,run_number,step_size,howlong):
        frames = []
        for name in self.inputs:
            if not name.startswith('df'):
                continue
            df = self.get_input_slot(name).data()
            df = df[df.columns.difference([self.UPDATE_COLUMN])]
            frames.append(df)
        df = frames[0]
        for other in frames[1:]:
            df = pd.merge(df, other, **self.merge_kwds)
        df[self.UPDATE_COLUMN] = run_number
        self._df = df
        return self._return_run_step(self.state_blocked, steps_run=len(self._df))
