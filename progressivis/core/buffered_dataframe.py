import pandas as pd
import numpy as np

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

    def df(self):
        return self._df

    def resize(self, l):
        lb = 0 if self._base is None else len(self._base)
        if l > lb:
            n = next_pow2(l)
            to_add = n-lb
            if lb==0:
                self._base = pd.DataFrame([],index=range(0,n))
            else:
                self._base = self._base.append(pd.DataFrame({},index=range(lb,n)))
        self._df = self._base.loc[0:l-1]
        return self._df

    def append(self, df):
        if df is None or len(df)==0:
            return
        if self._base is None:
            n = next_pow2(len(df))
            to_add = n-len(df)
            self._base = df.append(pd.DataFrame([],index=range(len(df),n)))
            self._df = self._base.loc[0:len(df)-1]
        else:
            start=len(self._df)
            end  =len(df)+len(self._df)
            self.resize(end)
            df.index = range(start,end)
            self._df.loc[start:end-1] = df
