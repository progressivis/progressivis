from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor
from .nary import NAry 

import pandas as pd

class CombineFirst(NAry):
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
            df = df.combine_first(other)
        df[self.UPDATE_COLUMN] = run_number
        self._df = df
        return self._return_run_step(self.state_blocked, steps_run=len(self._df))
