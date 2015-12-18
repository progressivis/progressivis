import pandas as pd

import logging
logger = logging.getLogger(__name__)


# See view-source:http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2Float
def next_pow2(v):
  v -= 1;
  v |= v >> 1
  v |= v >> 2
  v |= v >> 4
  v |= v >> 8
  v |= v >> 16
  v |= v >> 32
  return v+1


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

    def resize(self, l):
        lb = 0 if self._base is None else len(self._base)
        if l > lb:
            n = next_pow2(l)
            logger.info('Resizing dataframe %s from %d to %d', hex(id(self)), lb, n)
            if self._base is None:
                self._base = pd.DataFrame({},index=range(0,n))
            else:
                 # specifying the columns maintains the column order, otherwise, it gets sorted
                self._base = self._base.append(pd.DataFrame({},index=range(lb,n),
                                                            columns=self._base.columns))
                logger.debug('Dataframe %s grew to length=%d', hex(id(self)), len(self._base))
        self._df = self._base.iloc[0:l]
        return self._df

    def append(self, df): #TODO more work needed to handle, ignore_index=True):
        if df is None or len(df)==0:
            return
        if self._base is None:
            n = next_pow2(len(df))
            df.index = range(0,len(df))
            # specifying the columns maintains the column order, otherwise, it gets sorted
            self._base = df.append(pd.DataFrame([],index=range(len(df),n),columns=df.columns))
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
    
