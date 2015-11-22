from . import DataFrameModule

import pandas as pd

class Constant(DataFrameModule):
    def __init__(self, df, **kwds):        
        super(Constant, self).__init__(**kwds)
        assert df is None or isinstance(df, pd.DataFrame)
        if df is not None:
            df[self.UPDATE_COLUMN] = 1
        self._df = df

    def predict_step_size(self, duration):
        return 1
    
    def run_step(self,run_number,step_size,howlong):
        raise StopIteration()
