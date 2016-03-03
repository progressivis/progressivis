import pandas as pd

import logging
logger = logging.getLogger(__name__)

from .utils import next_pow2

if pd.__version__ > '0.18':
    def create_index(l,h):
        return pd.RangeIndex(l,h)
    def fix_index(df,l,h):
        df.index = pd.RangeIndex(l,h)
else:
    def create_index(l,h):
        return range(l,h)
    def fix_index(df,l,h):
        pass

class BufferedDataFrame(object):
    def __init__(self, df=None):
        self._df = None
        self._base = None
        self.append(df)

    def reset(self):
        self._df = None
        self._base = None

    def df(self):
        return self._df

    def _create_dataframe(self,index,columns,dtype):
        return pd.DataFrame({}, index=index,columns=columns,dtype=dtype)

    def resize(self, l):
        lb = 0 if self._base is None else len(self._base)
        if l > lb:
            n = next_pow2(l)
            logger.info('Resizing dataframe %s from %d to %d', hex(id(self)), lb, n)
            if self._base is None:
                self._base = self._create_dataframe(create_index(0,n),None,None)
            else:
                 # specifying the columns maintains the column order, otherwise, it gets sorted
                self._base = self._base.append(self._create_dataframe(create_index(lb,n),
                                                                      self._base.columns,
                                                                      self._base.dtypes))
                fix_index(self._base,0,n)
                logger.debug('Dataframe %s grew to length=%d', hex(id(self)), len(self._base))
        self._df = self._base.iloc[0:l]
        return self._df

    def append(self, df): #TODO more work needed to handle, ignore_index=True):
        if df is None or len(df)==0:
            return
        if self._base is None:
            n = next_pow2(len(df))
            # specifying the columns maintains the column order, otherwise, it gets sorted
            self._base = df.append(self._create_dataframe(create_index(len(df),n),df.columns,df.dtypes))
            fix_index(self._base, 0, n)
            self._df = self._base.iloc[0:len(df)]
        else:
            start=len(self._df)
            end  =len(df)+len(self._df)
            self.resize(end)
            self._df.iloc[start:end] = df.values
        return self

    def append_row(self, row):
        if self._base is None:
            df = pd.DataFrame(row, index=[0])
            self._base = df
            self._df = df
        else:
            start=len(self._df)
            end  = 1+len(self._df)
            self.resize(end)
            self._df.iloc[start] = row
        return self
    
