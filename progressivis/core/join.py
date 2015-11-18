from .nary import NAry

import pandas as pd

class Join(NAry):
    def __init__(self, **kwds):
        """Join(on=None, how='left', lsuffix='', rsuffix='',sort=False,id=None,scheduler=None,tracer=None,predictor=None,storage=None,input_descriptors=[],output_descriptors=[])
        """
        super(Join, self).__init__(**kwds)
        self.join_kwds = self._filter_kwds(kwds, pd.DataFrame.join)
        
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
            df = df.join(other, **self.join_kwds)
        df[self.UPDATE_COLUMN] = run_number
        self._df = df
        return self._return_run_step(self.state_blocked, steps_run=len(self._df))
